// Highlight the sub-section currently in view in the sidebar nav.
// The current page (section) is marked by Sphinx as `a.current`; this adds
// `nc-active` to the sidebar link of the heading you're reading (subsection).
(function () {
  function init() {
    var sidebar = document.querySelector(".sphinxsidebar");
    if (!sidebar) return;

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

    function update() {
      var current = entries[0];
      for (var i = 0; i < entries.length; i++) {
        if (entries[i].el.getBoundingClientRect().top <= 130) current = entries[i];
      }
      entries.forEach(function (e) {
        e.link.classList.toggle("nc-active", e === current);
      });
    }

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
