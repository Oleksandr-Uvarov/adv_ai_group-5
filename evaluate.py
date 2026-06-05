import copy
import time
import faulthandler

from stable_baselines3 import PPO

from env import GameEnv
from smallgridcnn import SmallGridCNN

faulthandler.enable()

# Seconds to pause between replayed steps (the per-step shot animation adds a
# little more on top of this). Recording itself runs with no rendering, so it
# is unaffected by this delay.
REPLAY_STEP_DELAY = 0.4


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
    """Label an episode from its final game state."""
    if truncated:
        return "truncated"
    if game.player_pos == game.exit_pos and game.has_key:
        return "won"
    return "lost"


def record_episodes(model, env, n_episodes):
    """Run the model for ``n_episodes`` with no rendering and capture the full
    per-step trajectory of each one. Every frame is a deep copy of the game
    state, so it can later be redrawn exactly as it happened. Returns a list of
    {"result": "won"|"lost"|"truncated", "frames": [game, ...]}."""
    episodes = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        # Snapshot the starting position, then one snapshot after every step.
        frames = [copy.deepcopy(env.game)]
        result = "truncated"

        for _ in range(env.game.step_limit):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            frames.append(copy.deepcopy(env.game))

            if done or truncated:
                result = classify_episode(env.game, truncated)
                break

        episodes.append({"result": result, "frames": frames})
        print(f"  recorded episode {i + 1}/{n_episodes}: {result}")

    return episodes


def watch_episodes(episodes, category, grid_size):
    """Replay every recorded episode matching ``category`` in a pygame window
    with a clickable 'Skip' button (or press the right-arrow / space key) to
    jump to the next one."""
    # Imported here so pure recording never needs pygame initialised.
    from pg.pygame_renderer import Renderer
    import pygame

    selected = [ep for ep in episodes if ep["result"] == category]
    if not selected:
        print(f"No '{category}' episodes were recorded.")
        return

    class ReplayRenderer(Renderer):
        """The normal renderer plus a button bar below the grid. The grid is
        drawn by the parent; we add the bar and turn the parent's hard-exit on
        QUIT into soft flags the replay loop can check."""

        BUTTON_HEIGHT = 64

        def __init__(self, grid_size=10):
            pygame.init()
            self.grid_size = grid_size
            self.width = grid_size * self.TILE_SIZE
            self.grid_height = grid_size * self.TILE_SIZE
            self.height = self.grid_height + self.BUTTON_HEIGHT
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("RL Roguelike — Replay")
            self.font = pygame.font.SysFont(None, 28)
            self._load_sprites()
            self.button_rect = pygame.Rect(
                0, self.grid_height, self.width, self.BUTTON_HEIGHT
            )
            self.label = ""
            self.skip_requested = False
            self.quit_requested = False

        def _draw_scene(self, game, extra=None, fireball=None):
            super()._draw_scene(game, extra=extra, fireball=fireball)
            self._draw_button()

        def _draw_button(self):
            hover = self.button_rect.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(self.screen, (20, 20, 20), self.button_rect)
            inner = self.button_rect.inflate(-12, -12)
            color = (70, 130, 180) if hover else (50, 90, 130)
            pygame.draw.rect(self.screen, color, inner, border_radius=8)
            text = self.font.render(
                f"{self.label}    |    Skip ▶  (→ / Space)",
                True, (255, 255, 255),
            )
            self.screen.blit(
                text,
                (inner.x + 15, inner.y + (inner.height - text.get_height()) // 2),
            )

        def _pump_events(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_requested = True
                elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                        and self.button_rect.collidepoint(event.pos)):
                    self.skip_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                        self.skip_requested = True
                    elif event.key == pygame.K_ESCAPE:
                        self.quit_requested = True

    renderer = ReplayRenderer(grid_size)
    print(f"Watching {len(selected)} '{category}' episode(s). "
          f"Close the window or press Esc to stop early.")

    for idx, episode in enumerate(selected):
        renderer.skip_requested = False
        renderer.label = f"{category}  {idx + 1}/{len(selected)}"

        for frame in episode["frames"]:
            time.sleep(REPLAY_STEP_DELAY)
            renderer.draw(frame)
            if renderer.quit_requested:
                renderer.close()
                return
            if renderer.skip_requested:
                break

        # Hold the final frame briefly so the outcome is visible before the
        # next episode starts (unless the user is skipping through).
        if not renderer.skip_requested:
            time.sleep(REPLAY_STEP_DELAY * 2)

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
    main("version_history/6_warlock_active_freeze_hp/zips/6_warlock_active_freeze_hp_ppo_3.zip", n_episodes=30)