document.addEventListener("DOMContentLoaded", () => {
  const landingPage = document.querySelector(
    "#welcome-to-isaac-lab-arena, #why-isaac-lab-arena",
  );
  if (!landingPage) {
    return;
  }

  // These two branded pages use a light-only visual system. Keep this choice
  // local to the current document so it does not overwrite a reader's theme
  // preference elsewhere in the documentation.
  document.documentElement.dataset.mode = "light";
  document.documentElement.dataset.theme = "light";

  document.querySelectorAll(".arena-agentic-giggles").forEach((panel) => {
    const tabs = Array.from(panel.querySelectorAll("[data-arena-agentic-tab]"));
    const examples = {
      domestic: panel.querySelector(".arena-agentic-example-domestic"),
      industrial: panel.querySelector(".arena-agentic-example-industrial"),
    };

    const selectExample = (name) => {
      tabs.forEach((tab) => {
        const selected = tab.dataset.arenaAgenticTab === name;
        tab.classList.toggle("arena-agentic-tab-active", selected);
        tab.setAttribute("aria-pressed", String(selected));
      });

      Object.entries(examples).forEach(([key, example]) => {
        if (example) {
          example.classList.toggle("arena-agentic-example-active", key === name);
        }
      });
    };

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => selectExample(tab.dataset.arenaAgenticTab));
    });
  });

  const videos = Array.from(document.querySelectorAll("video"));
  const videoLabels = {
    "relational-placement-solver.mp4": "Placement solver resolving spatial relationships",
    "relational-placement-resolved.mp4": "Simulation environment built from the resolved placement",
    "hdr_web.mp4": "HDR background variation",
    "color_web.mp4": "Light color variation",
    "temperature_web.mp4": "Color temperature variation",
    "shadows_web.mp4": "Light direction variation",
    "big_pumpkin_in_bin_web.mp4": "Big pumpkin in bin evaluation environment",
    "mouse_on_keyboard_web.mp4": "Mouse on keyboard evaluation environment",
    "small_pumpkin_in_bin_web.mp4": "Small pumpkin in bin evaluation environment",
    "mustard_in_left_bin_web.mp4": "Mustard in left bin evaluation environment",
    "predicate-progress-rollouts.mp4": "Parallel rollouts with predicate status overlays",
  };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  videos.forEach((video) => {
    const source = video.querySelector("source")?.getAttribute("src");
    const filename = source?.split("/").pop();
    if (filename && videoLabels[filename]) {
      video.setAttribute("aria-label", videoLabels[filename]);
    }
    video.autoplay = false;
    video.preload = "none";
  });

  if (reduceMotion || !("IntersectionObserver" in window)) {
    videos.forEach((video) => {
      video.controls = true;
    });
    return;
  }

  const videoObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting) {
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    },
    { rootMargin: "96px 0px" },
  );

  videos.forEach((video) => videoObserver.observe(video));
});
