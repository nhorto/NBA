/* Stats tools (/lineups): a player can't be ticked both On and Off (ticking one clears the
   other), and range inputs with data-out="<output id>" print their value while dragging.
   The pages work without it (plain GET forms). */
(function () {
  "use strict";
  document.addEventListener("change", function (e) {
    var el = e.target;
    if (!el || el.type !== "checkbox" || !el.hasAttribute("data-pair") || !el.checked) return;
    var other = el.name === "on" ? "off" : "on";
    var form = el.form || document;
    form.querySelectorAll('input[name="' + other + '"][data-pair="' + el.getAttribute("data-pair") + '"]')
      .forEach(function (x) { x.checked = false; });
  });
  document.querySelectorAll("input[type=range][data-out]").forEach(function (r) {
    var out = document.getElementById(r.getAttribute("data-out"));
    if (!out) return;
    r.addEventListener("input", function () { out.textContent = r.value; });
  });
})();
