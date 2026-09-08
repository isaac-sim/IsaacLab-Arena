document.addEventListener("DOMContentLoaded", () => {
  const landingPage = document.querySelector(
    "#welcome-to-isaac-lab-arena, #why-isaac-lab-arena",
  );
  if (!landingPage) {
    return;
  }

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
});
