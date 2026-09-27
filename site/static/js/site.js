// Small progressive enhancements: theme toggle, table of contents, and copy buttons.
// Every page works without this file.
(function () {
  "use strict";

  var root = document.documentElement;
  var prefersDark = window.matchMedia("(prefers-color-scheme: dark)");

  function effectiveTheme() {
    return root.getAttribute("data-theme") || (prefersDark.matches ? "dark" : "light");
  }

  function setupThemeToggle() {
    var button = document.querySelector(".theme-toggle");
    if (!button) return;
    function label() {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      button.setAttribute("aria-label", "Switch to " + next + " theme");
      button.setAttribute("title", "Switch to " + next + " theme");
    }
    button.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage may be blocked */ }
      label();
    });
    if (prefersDark.addEventListener) prefersDark.addEventListener("change", label);
    label();
  }

  function setupTableOfContents() {
    var details = document.querySelector(".toc-details");
    if (!details) return;
    var wide = window.matchMedia("(min-width: 68rem)");
    function sync() { details.open = wide.matches; }
    sync();
    if (wide.addEventListener) wide.addEventListener("change", sync);
    details.addEventListener("click", function (event) {
      if (!wide.matches && event.target.closest("a")) details.open = false;
    });

    // Highlight the section currently being read.
    var links = [];
    details.querySelectorAll('a[href^="#"]').forEach(function (link) {
      var target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      if (target) links.push({ link: link, target: target });
    });
    if (!links.length) return;
    var active = null;
    var pending = false;
    function update() {
      pending = false;
      var offset = parseFloat(getComputedStyle(root).scrollPaddingTop) || 80;
      var current = links[0];
      for (var i = 0; i < links.length; i++) {
        if (links[i].target.getBoundingClientRect().top - offset <= 8) current = links[i];
        else break;
      }
      if (current === active) return;
      if (active) active.link.classList.remove("is-active");
      current.link.classList.add("is-active");
      active = current;
    }
    window.addEventListener("scroll", function () {
      if (!pending) { pending = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    update();
  }

  function setupCopyButtons() {
    if (!navigator.clipboard) return;
    document.querySelectorAll(".code-block").forEach(function (block) {
      var code = block.querySelector("pre code");
      if (!code) return;
      var button = document.createElement("button");
      button.type = "button";
      button.className = "copy-button";
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy code to clipboard");
      button.addEventListener("click", function () {
        navigator.clipboard.writeText(code.textContent.replace(/\n$/, "")).then(function () {
          button.textContent = "Copied";
          button.classList.add("is-copied");
        }, function () {
          button.textContent = "Copy failed";
        });
        window.setTimeout(function () {
          button.textContent = "Copy";
          button.classList.remove("is-copied");
        }, 1800);
      });
      block.appendChild(button);
    });
  }

  setupThemeToggle();
  setupTableOfContents();
  setupCopyButtons();
})();
