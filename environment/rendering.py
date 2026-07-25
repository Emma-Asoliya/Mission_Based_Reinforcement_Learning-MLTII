"""
TriageRenderer
==============
A pygame dashboard that visualizes the MentalHealthTriageEnv in real time:
- Four resource panels (Self-Help, Peer, Professional, Crisis) showing
  live capacity bars and occupied slot icons.
- The current incoming request shown as a colored pulse (color = urgency).
- A live queue panel showing pending requests, coored by urgency.
- A HUD showing step count, resolved count, and running reward for the episode.
"""

import pygame 
WIDTH, HEIGHT = 900, 560
BG_COLOR = (18, 20, 26)
PANEL_COLOR = (30, 34, 44)
TEXT_COLOR = (235, 235, 240)
ACCENT = (90, 160, 255)

URGENCY_COLORS = {
    0: (90, 200, 130),
    1: (240, 200, 80),
    2: (240, 140, 60),
    3: (230, 70, 70),
}

RESOURCE_NAMES = ["Self-Help", "Peer Counselor", "Professional", "Crisis"]

class TriageRenderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Mental Health Support Triage : Agent Dashboard")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 14)
        self.font_big = pygame.font.SysFont("Arial", 24, bold=True)

    def render(self, env):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill(BG_COLOR)
        self._draw_header(env)
        self._draw_resource_panels(env)
        self._draw_current_request(env)
        self._draw_queue_panel(env)

        pygame.display.flip()
        self.clock.tick(4)

    def _draw_header(self, env):
        title = self.font_big.render("Mental Health Support Triage", True, TEXT_COLOR)
        self.screen.blit(title, (20, 15))
        stats = (f"Step: {env.time_step}/{env.max_steps}  "
                 f"Resolved {env.requests_resolved}/{env.max_requests}  "
                 f"Episode Reward {env.episode_reward:.1f}")
        stat_surf = self.font_small.render(stats, True, (170, 170, 180))
        self.screen.blit(stat_surf, (20, 45))

    def _draw_resource_panels(self, env):
        panel_w, panel_h = 190, 300
        gap = 20
        start_x = 20
        y = 90
        from environment.custom_env import MAX_CAPACITY

        for i, name in enumerate(RESOURCE_NAMES):
            x = start_x + i * (panel_w + gap)
            rect = pygame.Rect(x, y, panel_w, panel_h)
            pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=10)
            pygame.draw.rect(self.screen, ACCENT, rect, width=2, border_radius=10)

            label = self.font.render(name, True, TEXT_COLOR)
            self.screen.blit(label, (x + 10, y + 10))

            occupied = len(env.occupied[i])
            capacity = int(MAX_CAPACITY[i])
            cap_text = self.font_small.render(f"{occupied}/{capacity} in session",
                                              True, (180, 180, 190))
            self.screen.blit(cap_text, (x + 10, y +38))

            bar_x, bar_y, bar_w, bar_h = x + 10, y + 62, panel_w - 20, 16
            pygame.draw.rect(self.screen, (50, 54, 64), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            fill_ratio = occupied / max(capacity, 1)
            fill_w = int(bar_w * fill_ratio)
            fill_color = (230, 90, 90) if fill_ratio > 0.8 else ACCENT
            pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_w, bar_h), border_radius=4)


            icon_y = y + 95
            for s in range(min(capacity, 12)):
                icon_x = x + 15 + (s % 4) * 40
                icon_yy = icon_y + (s // 4) * 40
                color = ACCENT if s < occupied else (50, 54, 64)
                pygame.draw.circle(self.screen, color, (icon_x, icon_yy), 12)

    def _draw_current_request(self, env):
            y = 410
            rect = pygame.Rect(20, y, 860, 60)
            pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=10)

            req = env.current_request
            if req is None:
                txt = self.font.render("No active request", True, (150, 150, 160))
                self.screen.blit(txt, (35, y + 18))
                return

            color = URGENCY_COLORS[req["urgency"]]
            pygame.draw.circle(self.screen, color, (50, y + 30), 16)

            from environment.custom_env import CATEGORY_NAMES
            urgency_label = ["Low", "Moderate", "High", "CRISIS"][req["urgency"]]
            txt = self.font.render(
                f"Current request  | Urgency: {urgency_label}  |  "
                f"Category: {CATEGORY_NAMES[req['category']]}  |  Waiting: {req['wait']} steps",
                True, TEXT_COLOR)
            self.screen.blit(txt, (80, y + 20))

    def _draw_queue_panel(self, env):
            y = 480
            rect = pygame.Rect (20, y, 860, 60)
            pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=10)
            label = self.font_small.render(f"Pending queue: {len(env.pending)} waiting", True, TEXT_COLOR)
            self.screen.blit(label, (35, y + 8))

            x = 35
            for r in env.pending[:20]:
                color = URGENCY_COLORS[r["urgency"]]
                pygame.draw.circle(self.screen, color, (x, y +38), 8)
                x += 22

    def close(self):
                pygame.quit()
