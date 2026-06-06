import copy
import faulthandler

from stable_baselines3 import PPO

from env import GameEnv
from smallgridcnn import SmallGridCNN

faulthandler.enable()

# Seconds between auto-advanced frames while the replay is playing. Recording
# itself runs with no rendering, so it is unaffected by this delay.
REPLAY_STEP_DELAY = 0.4

# The glyph shown for the agent's last action in the replay HUD. Movement is
# WASD; any of the four shoot actions shows "S"; freeze shows "F". (Action ids
# come from Game.step: 0=up 1=down 2=left 3=right 4-7=shoot 8=freeze.)
ACTION_GLYPH = {0: "w", 1: "s", 2: "a", 3: "d",
                4: "S", 5: "S", 6: "S", 7: "S", 8: "F"}


def load_model(model_path, env):
    policy_kwargs = dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    return PPO.load(
        model_path,
        env=env,
        custom_objects={
            "device": "cpu",
            "policy_kwargs": policy_kwargs,
        },
    )


def classify_episode(game, truncated):
    """Label an episode from its final game state. Uses the explicit ``won`` flag
    rather than position+key: now that the exit only opens once every enemy is
    dead, a player can die while standing on the exit holding the key, which is a
    loss, not a win."""
    if game.won:
        return "won"
    if truncated:
        return "truncated"
    return "lost"


def evaluate(model_path, n_episodes=10000, pygame_overview=False, grid_size=10):
    """Run a trained model headlessly for ``n_episodes`` and tally the outcomes.

    Returns ``{"n_episodes", "n_won", "n_lost", "n_truncated"}``, the shape
    ``version_utils.write_version_file`` expects for its ``eval_results``. This is
    the function train.py / modal_train.py call right after training; it does no
    rendering (``pygame_overview`` is accepted for backwards compatibility but the
    tally is always headless - use ``main()`` to actually watch episodes)."""
    env = GameEnv(grid_size=grid_size, render_mode=None)
    model = load_model(model_path, env)

    counts = {"won": 0, "lost": 0, "truncated": 0}
    for _ in range(n_episodes):
        obs, _ = env.reset()
        result = "truncated"
        # An episode now spans up to N_LEVELS levels, each with its own step
        # budget, so allow that many steps before giving up on the episode.
        for _ in range(env.game.step_limit * env.game.N_LEVELS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                result = classify_episode(env.game, truncated)
                break
        counts[result] += 1
    env.close()

    return {
        "n_episodes": n_episodes,
        "n_won": counts["won"],
        "n_lost": counts["lost"],
        "n_truncated": counts["truncated"],
    }


def record_episodes(model, env, n_episodes):
    """Run the model for ``n_episodes`` with no rendering and capture the full
    per-step trajectory of each one. Every frame is a deep copy of the game
    state, so it can later be redrawn exactly as it happened. Returns a list of
    {"result": "won"|"lost"|"truncated", "frames": [game, ...]}."""
    episodes = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        # Tag each frame with the action that produced it so the replay HUD can
        # show the agent's last move. The starting frame has no preceding action.
        env.game.last_action = None
        # Snapshot the starting position, then one snapshot after every step.
        frames = [copy.deepcopy(env.game)]
        result = "truncated"

        # An episode now spans up to N_LEVELS levels, each with its own step
        # budget, so allow that many steps before giving up on the episode.
        for _ in range(env.game.step_limit * env.game.N_LEVELS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            env.game.last_action = int(action)
            frames.append(copy.deepcopy(env.game))

            if done or truncated:
                result = classify_episode(env.game, truncated)
                break

        episodes.append({"result": result, "frames": frames})
        print(f"  recorded episode {i + 1}/{n_episodes}: {result}")

    return episodes


def watch_episodes(episodes, category, grid_size):
    """Replay every recorded episode matching ``category`` in a pygame window
    with a control panel: an HP bar, the agent's last move, a freeze indicator,
    and Play/Pause + Prev/Next + Skip controls. The loop is tick-based (not
    sleep-based) so pausing and stepping stay responsive.

    Controls: Space play/pause, Left/Right step a frame (pauses), N skip to the
    next episode, Esc quit. The same actions are available as on-screen buttons."""
    # Imported here so pure recording never needs pygame initialised.
    from pg.pygame_renderer import Renderer
    import pygame

    selected = [ep for ep in episodes if ep["result"] == category]
    if not selected:
        print(f"No '{category}' episodes were recorded.")
        return

    class ReplayRenderer(Renderer):
        """The normal grid renderer plus a control panel below it. The grid and
        entities are drawn by the parent's _draw_scene; we add the HUD and the
        buttons, draw a static frame (no blocking shot animation, so stepping is
        instant), and turn the parent's hard-exit on QUIT into soft flags the
        replay loop checks."""

        PANEL_HEIGHT = 140

        def __init__(self, grid_size=10):
            pygame.init()
            self.grid_size = grid_size
            self.width = grid_size * self.TILE_SIZE
            self.grid_height = grid_size * self.TILE_SIZE
            self.height = self.grid_height + self.PANEL_HEIGHT
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("RL Roguelike - Replay")
            self.font = pygame.font.SysFont(None, 28)
            self.small_font = pygame.font.SysFont(None, 22)
            self._load_sprites()

            self.panel_rect = pygame.Rect(
                0, self.grid_height, self.width, self.PANEL_HEIGHT
            )
            self._build_buttons()

            self.ep_label = ""
            # Time (ms) the current frame began showing, so the player's shot can
            # be animated tile-by-tile from when the frame appears.
            self.anim_start_ms = 0
            self.reset_requests()

        # --- control state -------------------------------------------------
        def reset_requests(self):
            self.quit_requested = False
            self.skip_requested = False
            self.toggle_requested = False
            self.prev_requested = False
            self.next_requested = False

        def _build_buttons(self):
            margin, gap, btn_h = 12, 10, 50
            names = ["prev", "playpause", "next", "skip"]
            btn_w = (self.width - 2 * margin - (len(names) - 1) * gap) // len(names)
            y = self.grid_height + self.PANEL_HEIGHT - btn_h - 8
            self.buttons = {}
            x = margin
            for name in names:
                self.buttons[name] = pygame.Rect(x, y, btn_w, btn_h)
                x += btn_w + gap

        # --- drawing -------------------------------------------------------
        def render(self, game, frame_idx, n_frames, playing):
            self._draw_scene(game)          # parent: grid + entities + spikes
            self._draw_shot(game)           # static beam for a shoot frame
            self._draw_panel(game, frame_idx, n_frames, playing)
            pygame.display.flip()

        def _draw_shot(self, game):
            # Animate the player's shot as a single projectile flying one tile at
            # a time (SHOT_FRAME_MS per tile) from when the frame appeared, rather
            # than lighting up the whole path at once. Once it has travelled past
            # the last tile it is gone (the shot is hitscan; the target is already
            # removed from this frame's state).
            shot = getattr(game, "last_shot", None)
            if not shot or not shot.get("path"):
                return
            path = shot["path"]
            idx = (pygame.time.get_ticks() - self.anim_start_ms) // self.SHOT_FRAME_MS
            if idx >= len(path):
                return
            r, c = path[idx]
            ts = self.TILE_SIZE
            self.screen.blit(self.sprites["fireball"], (c * ts, r * ts))

        def _draw_panel(self, game, frame_idx, n_frames, playing):
            pygame.draw.rect(self.screen, (20, 20, 20), self.panel_rect)
            self._draw_hud(game, frame_idx, n_frames)
            self._draw_buttons(playing)
            hint = "Space: play/pause    <- -> : step    N: skip    Esc: quit"
            self.screen.blit(
                self.small_font.render(hint, True, (140, 140, 140)),
                (12, self.grid_height + 58),
            )

        def _draw_hud(self, game, frame_idx, n_frames):
            m = 12
            top = self.grid_height

            # HP bar, scaled to current HP and coloured by how much is left.
            hp = max(0, game.hp)
            ratio = hp / game.MAX_HP if game.MAX_HP else 0
            self.screen.blit(
                self.small_font.render(f"HP {hp}/{game.MAX_HP}", True, (230, 230, 230)),
                (m, top + 6),
            )
            bar = pygame.Rect(m, top + 28, 200, 16)
            pygame.draw.rect(self.screen, (60, 60, 60), bar)
            fill = pygame.Rect(bar.x, bar.y, int(bar.width * ratio), bar.height)
            color = ((80, 200, 90) if ratio > 0.5
                     else (220, 200, 70) if ratio > 0.25 else (220, 70, 70))
            pygame.draw.rect(self.screen, color, fill)
            pygame.draw.rect(self.screen, (180, 180, 180), bar, 1)

            # Last move glyph (WASD / S shoot / F freeze).
            glyph = ACTION_GLYPH.get(getattr(game, "last_action", None), "-")
            self.screen.blit(
                self.font.render(f"Move: {glyph}", True, (230, 230, 230)),
                (m + 230, top + 8),
            )

            # Freeze charges remaining (persist across levels); red once empty.
            charges = getattr(game, "freeze_charges", 0)
            if charges > 0:
                ftext, fcolor = f"Freeze x{charges}", (200, 200, 200)
            else:
                ftext, fcolor = "Freeze: none", (235, 70, 70)
            self.screen.blit(self.small_font.render(ftext, True, fcolor), (m + 230, top + 36))

            # Episode label on the right, with the level + frame counter below it.
            lab = self.small_font.render(self.ep_label, True, (230, 230, 230))
            self.screen.blit(lab, (self.width - lab.get_width() - m, top + 6))
            lvl = getattr(game, "level", 0)
            n_levels = getattr(game, "N_LEVELS", 1)
            fc = self.small_font.render(
                f"Level {lvl + 1}/{n_levels}   Frame {frame_idx + 1}/{n_frames}",
                True, (200, 200, 200))
            self.screen.blit(fc, (self.width - fc.get_width() - m, top + 28))

        def _draw_buttons(self, playing):
            mouse = pygame.mouse.get_pos()
            labels = {
                "prev": "<< Prev",
                "playpause": "Pause" if playing else "Play",
                "next": "Next >>",
                "skip": "Skip",
            }
            for name, rect in self.buttons.items():
                hover = rect.collidepoint(mouse)
                pygame.draw.rect(
                    self.screen, (70, 130, 180) if hover else (50, 90, 130),
                    rect, border_radius=8,
                )
                t = self.font.render(labels[name], True, (255, 255, 255))
                self.screen.blit(
                    t, (rect.centerx - t.get_width() // 2,
                        rect.centery - t.get_height() // 2),
                )

        # --- events --------------------------------------------------------
        def _pump_events(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_requested = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if self.buttons["prev"].collidepoint(pos):
                        self.prev_requested = True
                    elif self.buttons["next"].collidepoint(pos):
                        self.next_requested = True
                    elif self.buttons["playpause"].collidepoint(pos):
                        self.toggle_requested = True
                    elif self.buttons["skip"].collidepoint(pos):
                        self.skip_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_requested = True
                    elif event.key == pygame.K_LEFT:
                        self.prev_requested = True
                    elif event.key == pygame.K_RIGHT:
                        self.next_requested = True
                    elif event.key in (pygame.K_n, pygame.K_TAB):
                        self.skip_requested = True
                    elif event.key == pygame.K_ESCAPE:
                        self.quit_requested = True

    renderer = ReplayRenderer(grid_size)
    clock = pygame.time.Clock()
    step_ms = int(REPLAY_STEP_DELAY * 1000)
    end_hold_ms = max(2 * step_ms, 800)  # linger on the last frame before auto-advancing
    print(f"Watching {len(selected)} '{category}' episode(s). "
          f"Space play/pause, arrows step, N skip, Esc quit.")

    for idx, episode in enumerate(selected):
        frames = episode["frames"]
        renderer.ep_label = f"{category}  {idx + 1}/{len(selected)}"
        renderer.reset_requests()
        frame_idx = 0
        shown_idx = -1
        playing = True
        last_advance = pygame.time.get_ticks()
        advance_to_next = False

        while True:
            renderer._pump_events()
            now = pygame.time.get_ticks()

            if renderer.quit_requested:
                renderer.close()
                return
            if renderer.skip_requested:
                break
            if renderer.toggle_requested:
                playing = not playing
                if playing:
                    last_advance = now  # so we don't instantly trip the end-hold
            if renderer.prev_requested:
                playing = False
                frame_idx = max(0, frame_idx - 1)
            if renderer.next_requested:
                playing = False
                frame_idx = min(len(frames) - 1, frame_idx + 1)
            # Clear the per-iteration (consumable) requests; quit/skip already
            # caused a return/break above.
            renderer.toggle_requested = False
            renderer.prev_requested = False
            renderer.next_requested = False

            if playing:
                if frame_idx < len(frames) - 1:
                    if now - last_advance >= step_ms:
                        frame_idx += 1
                        last_advance = now
                elif now - last_advance >= end_hold_ms:
                    # Reached the end while still playing: roll on to the next
                    # episode automatically (staying in play mode) instead of
                    # waiting for a manual Skip.
                    advance_to_next = True

            # Restart the shot fly-animation whenever the displayed frame changes.
            if frame_idx != shown_idx:
                shown_idx = frame_idx
                renderer.anim_start_ms = now

            renderer.render(frames[frame_idx], frame_idx, len(frames), playing)
            clock.tick(60)

            if advance_to_next:
                break

    renderer.close()


def main(model_path, n_episodes=30, grid_size=10):
    # Recording uses no renderer so it runs as fast as the model can predict.
    env = GameEnv(grid_size=grid_size, render_mode=None)
    model = load_model(model_path, env)

    print(f"Recording {n_episodes} episode(s)...")
    episodes = record_episodes(model, env, n_episodes)
    env.close()

    counts = {"won": 0, "lost": 0, "truncated": 0}
    for episode in episodes:
        counts[episode["result"]] += 1
    print("\nDone recording. Results:")
    print(f"  won:       {counts['won']}")
    print(f"  lost:      {counts['lost']}")
    print(f"  truncated: {counts['truncated']}")

    while True:
        choice = input(
            "\nWatch which episodes? [won/lost/truncated] (q to quit): "
        ).strip().lower()
        if choice in ("q", "quit", "exit", ""):
            break
        if choice not in ("won", "lost", "truncated"):
            print("Please type 'won', 'lost', 'truncated', or 'q'.")
            continue
        watch_episodes(episodes, choice, grid_size)


if __name__ == "__main__":
    main("version_history/9_levels_and_potion/zips/9_levels_and_potion_ppo_3_20260606-111428-873708.zip", n_episodes=30)