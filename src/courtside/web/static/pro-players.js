/* Player page: the reference line of the "Recent games" chart (stats_player.html, data from
   stats/player_tools.py). Vanilla, no dependencies; loaded only on /player/<id>.

   The server renders everything (bars, default line, summary), so the chart reads correctly
   without JS. This file only lets the visitor move the line: the number input and the slider
   stay in sync, bars at or above the line get class "hit" (solid fill + bold value; misses are
   outlined), the line and its label move, and the summary text is recomputed with the same
   wording and rounding as player_tools.summary(). The URL's ?line= follows (replaceState) so
   the view can be shared. Data: <script type="application/json" id="pl2-data">
   {stat, label, integer, line, default, ymax, top, bot, parts: [{label, values}]}. */
(function () {
  "use strict";
  var doc = document;

  function init() {
    var el = doc.getElementById("pl2-data");
    if (!el) return;
    var d;
    try { d = JSON.parse(el.textContent); } catch (e) { return; }
    var svg = doc.querySelector(".pl2-chart");
    var num = doc.getElementById("pl2-line-n");
    var rng = doc.getElementById("pl2-line-r");
    var sum = doc.getElementById("pl2-summary");
    if (!svg || !num || !sum) return;
    var bars = svg.querySelectorAll(".pl2-bar");
    var lineG = svg.querySelector(".pl2-line");
    var lineT = lineG && lineG.querySelector("text");
    var form = num.form;
    var EPS = 1e-9;

    function th(line) {
      return d.integer ? String(Math.ceil(line - EPS)) : String(+line.toFixed(1));
    }
    function pct(k, n) { return n ? Math.floor(100 * k / n + 0.5) + "%" : "–"; }
    function hits(vals, line) {
      var k = 0;
      for (var i = 0; i < vals.length; i++) if (vals[i] >= line - EPS) k++;
      return k;
    }
    function summary(line) {
      var t = th(line), out = [];
      d.parts.forEach(function (p, i) {
        var n = p.values.length, k = hits(p.values, line);
        if (i === 0) out.push(n ? "Reached " + t + "+ in " + k + " of " + p.label + " (" + pct(k, n) + ")"
                                : "No games for " + p.label);
        else if (n) out.push(p.label + " " + k + " of " + n + " (" + pct(k, n) + ")");
      });
      return out.join(" · ");
    }
    function y(v) {
      return Math.round((d.bot - (d.bot - d.top) * Math.min(Math.max(v, 0), d.ymax) / d.ymax) * 10) / 10;
    }
    function apply(line, from) {
      if (!isFinite(line) || line < 0) return;
      for (var i = 0; i < bars.length; i++) {
        var v = parseFloat(bars[i].getAttribute("data-v"));
        bars[i].classList.toggle("hit", v >= line - EPS);
      }
      if (lineG) lineG.setAttribute("transform", "translate(0 " + y(line) + ")");
      if (lineT) lineT.textContent = th(line) + "+";
      sum.textContent = summary(line);
      if (from !== num) num.value = line;
      if (rng && from !== rng) rng.value = Math.min(line, parseFloat(rng.max) || line);
      try {
        var u = new URL(window.location.href);
        if (Math.abs(line - d["default"]) < EPS) u.searchParams.delete("line");
        else u.searchParams.set("line", String(line));
        window.history.replaceState(null, "", u.pathname + u.search + u.hash);
      } catch (e) { /* old browsers: the form still works */ }
    }

    num.addEventListener("input", function () {
      var v = parseFloat(num.value);
      if (isFinite(v)) apply(v, num);
    });
    if (rng) rng.addEventListener("input", function () { apply(parseFloat(rng.value), rng); });
    if (form) form.addEventListener("submit", function (e) {
      e.preventDefault();
      var v = parseFloat(num.value);
      if (isFinite(v)) apply(v, null);
    });
    var reset = doc.querySelector(".pl2-reset[data-line]");
    if (reset) reset.addEventListener("click", function (e) {
      e.preventDefault();
      apply(parseFloat(reset.getAttribute("data-line")), null);
    });
    /* click a bar (or Enter / Space on a focused one) to put the line at that game's value */
    svg.addEventListener("click", function (e) {
      var g = e.target.closest && e.target.closest(".pl2-bar");
      if (g) apply(parseFloat(g.getAttribute("data-v")), null);
    });
    svg.addEventListener("keydown", function (e) {
      var g = e.target.closest && e.target.closest(".pl2-bar");
      if (g && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault();
        apply(parseFloat(g.getAttribute("data-v")), null);
      }
    });
  }

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", init);
  else init();
})();
