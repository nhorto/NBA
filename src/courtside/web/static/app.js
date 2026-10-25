/* Courtside: tabs, open-on-hash, active section in the section nav. No dependencies. */
(function () {
  "use strict";
  var root = document.documentElement;

  // ---- tabs (ARIA tablist; arrow keys move between tabs)
  document.querySelectorAll("[data-tabs]").forEach(function (box) {
    var tabs = Array.prototype.slice.call(box.querySelectorAll('[role="tab"]'));
    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        var p = document.getElementById(t.getAttribute("aria-controls"));
        if (p) p.hidden = !on;
      });
      if (focus) tab.focus();
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(t, false); });
      t.addEventListener("keydown", function (e) {
        var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
        if (e.key === "Home") j = 0;
        if (e.key === "End") j = tabs.length - 1;
        if (j === null) return;
        e.preventDefault();
        select(tabs[(j + tabs.length) % tabs.length], true);
      });
    });
  });

  // ---- "Copy lineup" buttons (DFS page): shown only when the clipboard API exists
  document.querySelectorAll("[data-copy]").forEach(function (b) {
    var src = document.querySelector(b.getAttribute("data-copy"));
    if (!src || !navigator.clipboard) return;
    b.hidden = false;
    b.addEventListener("click", function () {
      navigator.clipboard.writeText(src.textContent).then(function () {
        var t = b.textContent; b.textContent = "Copied";
        setTimeout(function () { b.textContent = t; }, 1600);
      });
    });
  });

  // ---- a link to a collapsed row (e.g. a "Decided early" game) opens it
  function openTarget() {
    var id = location.hash.slice(1);
    var el = id && document.getElementById(id);
    if (el && el.tagName === "DETAILS") el.open = true;
  }
  window.addEventListener("hashchange", openTarget);
  openTarget();

  // ---- highlight the section in view
  var links = document.querySelectorAll(".secnav a[href^='#']");
  if (links.length && "IntersectionObserver" in window) {
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var current = null;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          if (current) current.classList.remove("is-active");
          current = map[en.target.id];
          if (current) {
            current.classList.add("is-active");
            var nav = current.parentElement;
            var l = current.offsetLeft - 16;
            if (l < nav.scrollLeft || current.offsetLeft + current.offsetWidth > nav.scrollLeft + nav.clientWidth) {
              nav.scrollTo({ left: l, behavior: "smooth" });
            }
          }
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) io.observe(el);
    });
  }
})();
