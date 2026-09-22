document.addEventListener("DOMContentLoaded", () => {
  const visualPage = document.querySelector(
    "#welcome-to-isaac-lab-arena, #why-isaac-lab-arena, #gallery",
  );
  if (!visualPage) {
    return;
  }

  // These branded pages use a light-only visual system. Keep this choice
  // local to the current document so other documentation retains the
  // reader's theme preference.
  document.documentElement.dataset.mode = "light";
  document.documentElement.dataset.theme = "light";

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
    "alphabet_soup_can_web.mp4": "Object swap evaluation with an alphabet soup can",
    "billiard_hall_web.mp4": "Background swap evaluation in a billiard hall",
    "lemon_web.mp4": "Object swap evaluation with a lemon",
    "mug_web.mp4": "Object swap evaluation with a mug",
    "mustard_bottle_web.mp4": "Object swap evaluation with a mustard bottle",
    "orange_web.mp4": "Object swap evaluation with an orange",
    "rubiks_cube_home_office_web.mp4": "Object and background swap evaluation",
    "sugar_box_web.mp4": "Object swap evaluation with a sugar box",
    "tomato_sauce_can_web.mp4": "Object swap evaluation with a tomato sauce can",
    "bagels_on_plate_web.mp4": "Relational placement of bagels on a plate",
    "penisula_mustard_mesh_web.mp4": "Agent-generated mustard placement environment",
  };
  const videos = Array.from(visualPage.querySelectorAll("video"));
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
