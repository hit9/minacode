// Highlight the sub-section currently in view in the sidebar nav.
// The current page (section) is marked by Sphinx as `a.current`; this adds
// `nc-active` to the sidebar link of the heading you're reading (subsection).
(function () {
  function init() {
    var sidebar = document.querySelector(".sphinxsidebar");
    if (!sidebar) return;

    // Keep the reader's place in the long navigation tree across page loads.
    // sessionStorage scopes the position to this browser tab and documentation
    // site, while the guard keeps the script harmless when storage is blocked.
    try {
      var scrollKey = "wizolt-sidebar-scroll";
      var savedScroll = sessionStorage.getItem(scrollKey);
      if (savedScroll !== null) sidebar.scrollTop = Number(savedScroll);
      sidebar.addEventListener("scroll", function () {
        sessionStorage.setItem(scrollKey, String(sidebar.scrollTop));
      }, { passive: true });
    } catch (_) {
      // Highlighting still works without storage access.
    }

    var currentPage = sidebar.querySelector("a.current");
    var currentBranch = currentPage && currentPage.closest("li");
    if (!currentBranch) return;

    // The global table of contents uses fragment-only links for headings on
    // the current page. Restricting the search to its branch avoids matching
    // similarly named headings on other pages.
    var entries = [];
    currentBranch.querySelectorAll('a[href^="#"]').forEach(function (link) {
      var id = decodeURIComponent(link.hash.slice(1));
      var heading = id && document.getElementById(id);
      if (heading) entries.push({ el: heading, link: link });
    });
    if (!entries.length) return;

    var pinned = null;
    var pinTimer = 0;

    function activate(current) {
      entries.forEach(function (entry) {
        entry.link.classList.toggle("nc-active", entry === current);
      });
    }

    function pin(entry) {
      pinned = entry;
      activate(entry);
      clearTimeout(pinTimer);
      pinTimer = setTimeout(function () {
        pinned = null;
      }, 250);
    }

    function update() {
      if (pinned) return;
      var current = entries[0];
      for (var i = 0; i < entries.length; i++) {
        if (entries[i].el.getBoundingClientRect().top <= 32) current = entries[i];
      }
      activate(current);
    }

    entries.forEach(function (entry) {
      entry.link.addEventListener("click", function () {
        pin(entry);
      });
    });

    window.addEventListener("hashchange", function () {
      var entry = entries.find(function (candidate) {
        return candidate.link.hash === location.hash;
      });
      if (entry) pin(entry);
    });

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update, { passive: true });
    var initial = entries.find(function (entry) {
      return entry.link.hash === location.hash;
    });
    if (initial) pin(initial);
    else update();
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
