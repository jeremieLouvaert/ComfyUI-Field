// ComfyUI-Field -- FieldGradient "ramp" canvas widget.
//
// The node's `ramp` STRING widget (multiline) is the single source of truth:
// its value is JSON of the shape
//   {"version": 1, "stops": [{"p":0..1, "v":0..1, "i":"constant|linear|smooth"}, ...]}
// where each stop's `i` is the interpolation of the SEGMENT TO ITS RIGHT (so
// the last stop's `i` is inert). See nodes/field_gradient.py / utils/ramp.py
// and docs/field-phase2c-derivation.md section 2.5 for the server-side model
// this widget must stay bit-compatible with.
//
// This file adds a second, purely-visual CUSTOM widget above the STRING
// widget: a draggable-stop ramp editor. It owns nothing server-side -- on
// every edit it re-serialises the full stops array and writes it back into
// the STRING widget's `.value` (+ callback + setDirtyCanvas), so ComfyUI's
// normal serialize/queue path picks up the change exactly as if the user had
// typed it in by hand. The STRING widget stays visible below the canvas as
// the manual-JSON-paste escape hatch; editing it directly re-syncs the
// canvas via a hooked callback/inputEl listener.
//
// House pattern: WEB_DIRECTORY = "./web" (see __init__.py) + beforeRegisterNodeDef
// wrapping onNodeCreated for a specific node type by name (ComfyUI-Darkroom's
// darkroom_folder_picker.js), self-contained plain canvas/DOM (ComfyUI-EphemeralPeek).

import { app } from "../../scripts/app.js";

console.log("[FieldGradient] ramp widget v1");

// --- shared constants -------------------------------------------------------

const VALID_I = new Set(["constant", "linear", "smooth"]);
const DEFAULT_STOPS = [
  { p: 0.0, v: 0.0, i: "linear" },
  { p: 1.0, v: 1.0, i: "linear" },
];

const SIDE_PAD = 8;
const TOP_PAD = 8;
const STRIP_H = 70;
const CHIP_GAP = 6;
const CHIP_H = 16;
const BOTTOM_PAD = 6;
const HANDLE_R = 6;
const HANDLE_HIT_R = 10;
const REMOVE_THRESHOLD = 46; // px beyond strip top/bottom -> drag-out removal
const DUP_EPS_P = 0.006; // position-space "very near" epsilon for dedup snapping
const FINE_SCALE = 0.2; // shift-drag precision multiplier (delta-based)
const DBLCLICK_MS = 340;
const DBLCLICK_PX = 10;
const MAX_STOPS = 64;

function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}
function round4(x) {
  return Math.round(x * 10000) / 10000;
}
function quintic(u) {
  // 6u^5 - 15u^4 + 10u^3
  return u * u * u * (u * (u * 6 - 15) + 10);
}

// Approximates utils/ramp.py's eval_ramp: below first stop -> v[0], above
// last -> v[-1], else interpolate within the segment using the LEFT stop's
// `i`. `stops` must already be sorted by (p, _order) so that exact-position
// ties resolve "later wins", matching the server. This is a preview only
// (spec: "a reasonable visual approximation is fine").
function evalRampJS(t, stops) {
  const n = stops.length;
  if (n === 0) return 0;
  if (t <= stops[0].p) return stops[0].v;
  if (t >= stops[n - 1].p) return stops[n - 1].v;

  let segIdx = 0;
  for (let k = 0; k < n; k++) {
    if (stops[k].p <= t) segIdx = k;
    else break;
  }
  if (segIdx >= n - 1) return stops[n - 1].v;

  const a = stops[segIdx];
  const b = stops[segIdx + 1];
  const span = b.p - a.p;
  const u = span > 0 ? clamp01((t - a.p) / span) : 0;

  if (a.i === "constant") return a.v;
  if (a.i === "smooth") {
    const q = quintic(u);
    return (1 - q) * a.v + q * b.v;
  }
  return (1 - u) * a.v + u * b.v; // linear (also the fallback default)
}

function sortStopsCopy(stops) {
  return [...stops].sort((x, y) => x.p - y.p || x._order - y._order);
}

// --- the widget ---------------------------------------------------------

function createRampWidget(node, stringWidget) {
  const widget = {
    name: "ramp_canvas",
    type: "custom",
    value: "",
    options: { serialize: false }, // mirrors the STRING widget; never its own saved value

    // parsed state
    stops: [],
    envelope: null,
    orderCounter: 0,
    parseOk: true,
    errorMsg: "",
    _lastRawValue: undefined,

    // interaction state
    dragIndex: -1,
    dragRemoving: false,
    _dragLastPx: null,
    _dragLastPy: null,
    _lastClickTime: 0,
    _lastClickPos: null,

    // layout, captured each draw() so mouse() can hit-test the same geometry
    lastDrawY: 0,
    lastDrawW: 0,

    reparse() {
      const raw = stringWidget.value;
      this._lastRawValue = raw;
      try {
        const data = JSON.parse(raw);
        if (!data || typeof data !== "object" || Array.isArray(data)) {
          throw new Error("ramp JSON must be an object with 'version' and 'stops'");
        }
        if (!Array.isArray(data.stops) || data.stops.length === 0) {
          throw new Error("ramp.stops must be a non-empty array");
        }
        const withIdx = data.stops.map((s, idx) => ({ s, idx }));
        withIdx.sort((a, b) => Number(a.s.p) - Number(b.s.p) || a.idx - b.idx);
        const stops = withIdx.map((entry, order) => {
          const s = entry.s && typeof entry.s === "object" ? entry.s : {};
          const p = clamp01(Number(s.p));
          const v = clamp01(Number(s.v));
          if (!Number.isFinite(p) || !Number.isFinite(v)) {
            throw new Error(`stop[${entry.idx}] p/v is not a finite number`);
          }
          const i = VALID_I.has(s.i) ? s.i : "linear";
          return { ...s, p, v, i, _order: order };
        });
        this.envelope = { ...data, version: data.version === undefined ? 1 : data.version };
        this.stops = stops;
        this.orderCounter = stops.length;
        this.parseOk = true;
        this.errorMsg = "";
      } catch (err) {
        // Fail soft: keep the last good stops (or the 2-stop default) for
        // rendering only. Never touch stringWidget.value here -- the user's
        // string is left alone until they actually interact with the canvas.
        this.parseOk = false;
        this.errorMsg = String((err && err.message) || err);
        if (!this.stops || this.stops.length === 0) {
          this.stops = DEFAULT_STOPS.map((s, order) => ({ ...s, _order: order }));
          this.envelope = { version: 1 };
          this.orderCounter = this.stops.length;
        }
      }
    },

    // Builds the JSON-ready stops array (sorted, internal keys stripped) and
    // envelope without mutating this.stops's array order -- so it's safe to
    // call mid-drag without invalidating this.dragIndex.
    buildEnvelope() {
      const sorted = sortStopsCopy(this.stops);
      const outStops = sorted.map((s) => {
        const { _order, ...rest } = s;
        return { ...rest, p: round4(s.p), v: round4(s.v), i: s.i };
      });
      const base = this.envelope || {};
      return {
        ...base,
        version: base.version === undefined ? 1 : base.version,
        stops: outStops,
      };
    },

    // Writes the current state back to the STRING widget (the single source
    // of truth). `commitOrder`=true also reorders this.stops itself -- only
    // safe to do once a drag gesture has actually ended (dragIndex reset).
    writeString(commitOrder) {
      const envelope = this.buildEnvelope();
      this.envelope = envelope;
      if (commitOrder) this.stops = sortStopsCopy(this.stops);
      const str = JSON.stringify(envelope);
      this._lastRawValue = str;
      stringWidget.value = str;
      if (typeof stringWidget.callback === "function") {
        try {
          stringWidget.callback(str, app.canvas, node);
        } catch (e) {
          console.warn("[FieldGradient] ramp string callback threw", e);
        }
      }
      node.setDirtyCanvas(true, true);
    },

    computeSize(_width) {
      return [_width, TOP_PAD + STRIP_H + CHIP_GAP + CHIP_H + BOTTOM_PAD];
    },

    draw(ctx, node, widgetWidth, y, _widgetHeight) {
      try {
        if (stringWidget.value !== this._lastRawValue) this.reparse();
        this.lastDrawY = y;
        this.lastDrawW = widgetWidth;

        const x0 = SIDE_PAD;
        const w = Math.max(1, widgetWidth - SIDE_PAD * 2);
        const stripY = y + TOP_PAD;
        const chipY = stripY + STRIP_H + CHIP_GAP;

        ctx.save();

        ctx.fillStyle = "#1b1b1b";
        ctx.fillRect(x0, stripY, w, STRIP_H);

        ctx.strokeStyle = "#2a2a2a";
        ctx.beginPath();
        for (const f of [0.25, 0.5, 0.75]) {
          const gy = stripY + STRIP_H * (1 - f);
          ctx.moveTo(x0, gy);
          ctx.lineTo(x0 + w, gy);
        }
        ctx.stroke();

        const evalStops = sortStopsCopy(this.stops);

        // curve fill + stroke -- the evaluated ramp profile, p=0..1
        if (evalStops.length > 0) {
          const STEP = 2;
          ctx.beginPath();
          ctx.moveTo(x0, stripY + STRIP_H);
          for (let px = 0; px <= w; px += STEP) {
            const t = px / w;
            const val = clamp01(evalRampJS(t, evalStops));
            ctx.lineTo(x0 + px, stripY + STRIP_H * (1 - val));
          }
          ctx.lineTo(x0 + w, stripY + STRIP_H);
          ctx.closePath();
          ctx.fillStyle = "rgba(90,170,230,0.28)";
          ctx.fill();

          ctx.beginPath();
          for (let px = 0; px <= w; px += STEP) {
            const t = px / w;
            const val = clamp01(evalRampJS(t, evalStops));
            const py = stripY + STRIP_H * (1 - val);
            if (px === 0) ctx.moveTo(x0 + px, py);
            else ctx.lineTo(x0 + px, py);
          }
          ctx.strokeStyle = "#7cc4f0";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        ctx.strokeStyle = "#3a3a3a";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, stripY + 0.5, w - 1, STRIP_H - 1);

        if (!this.parseOk) {
          ctx.fillStyle = "rgba(90,20,20,0.85)";
          ctx.fillRect(x0, stripY, w, 14);
          ctx.fillStyle = "#f0a0a0";
          ctx.font = "9px monospace";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText("invalid ramp JSON -- fallback preview shown", x0 + 4, stripY + 7);
        }

        // handles + per-stop interpolation chips
        for (let idx = 0; idx < this.stops.length; idx++) {
          const s = this.stops[idx];
          const hx = x0 + s.p * w;
          const hy = stripY + STRIP_H * (1 - s.v);
          const isLast = idx === this.stops.length - 1;
          const dragging = this.dragIndex === idx;

          ctx.globalAlpha = dragging && this.dragRemoving ? 0.32 : 1.0;
          ctx.beginPath();
          ctx.fillStyle = dragging ? "#ffd166" : "#eaeaea";
          ctx.strokeStyle = "#111";
          ctx.lineWidth = 1;
          ctx.arc(hx, hy, HANDLE_R, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.globalAlpha = 1.0;

          const chipW = 18;
          const cx0 = Math.min(Math.max(hx - chipW / 2, x0), x0 + w - chipW);
          ctx.fillStyle = isLast ? "#262626" : "#33475a";
          ctx.fillRect(cx0, chipY, chipW, CHIP_H);
          ctx.strokeStyle = isLast ? "#3a3a3a" : "#4d6d8a";
          ctx.lineWidth = 1;
          ctx.strokeRect(cx0 + 0.5, chipY + 0.5, chipW - 1, CHIP_H - 1);
          ctx.fillStyle = isLast ? "#5c5c5c" : "#dfe8f0";
          ctx.font = "9px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          const label = s.i === "constant" ? "C" : s.i === "smooth" ? "S" : "L";
          ctx.fillText(label, cx0 + chipW / 2, chipY + CHIP_H / 2 + 1);

          // geometry cached for hit-testing in mouse()
          s._chipRect = { x: cx0, y: chipY, w: chipW, h: CHIP_H, disabled: isLast };
          s._handlePos = { x: hx, y: hy };
        }

        ctx.textAlign = "right";
        ctx.textBaseline = "alphabetic";
        ctx.font = "9px monospace";
        if (this.dragIndex !== -1 && this.stops[this.dragIndex]) {
          const s = this.stops[this.dragIndex];
          ctx.fillStyle = this.dragRemoving ? "#e08a8a" : "#9adca0";
          const txt = this.dragRemoving
            ? "release to remove"
            : `p ${s.p.toFixed(3)}  v ${s.v.toFixed(3)}  (shift = fine)`;
          ctx.fillText(txt, x0 + w, stripY - 3);
        } else {
          ctx.fillStyle = "#777";
          ctx.fillText(
            `${this.stops.length} stop${this.stops.length === 1 ? "" : "s"} -- dblclick to add, drag out to remove`,
            x0 + w,
            stripY - 3
          );
        }

        ctx.restore();
      } catch (err) {
        console.error("[FieldGradient] ramp widget draw() failed (rendering fallback):", err);
        try {
          ctx.save();
          ctx.fillStyle = "#4a1a1a";
          ctx.fillRect(SIDE_PAD, y + TOP_PAD, Math.max(1, widgetWidth - SIDE_PAD * 2), STRIP_H);
          ctx.fillStyle = "#f0a0a0";
          ctx.font = "10px monospace";
          ctx.fillText("[FieldGradient] ramp widget error -- see console", SIDE_PAD + 4, y + TOP_PAD + STRIP_H / 2);
          ctx.restore();
        } catch (_e2) {
          /* nothing more we can safely do */
        }
      }
    },

    mouse(event, pos, node) {
      try {
        if (!pos || !this.lastDrawW) return false;
        const x0 = SIDE_PAD;
        const w = Math.max(1, this.lastDrawW - SIDE_PAD * 2);
        const stripY = this.lastDrawY + TOP_PAD;
        const chipY = stripY + STRIP_H + CHIP_GAP;
        const px = pos[0];
        const py = pos[1];

        const t = event.type || "";
        const isDown = t.endsWith("down");
        const isMove = t.endsWith("move");
        const isUp = t.endsWith("up") || t === "click";

        if (isDown) {
          // 1. per-stop interpolation chip (cycle constant -> linear -> smooth)
          for (let idx = this.stops.length - 1; idx >= 0; idx--) {
            const r = this.stops[idx]._chipRect;
            if (r && px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) {
              if (!r.disabled) {
                const order = ["linear", "smooth", "constant"];
                const s = this.stops[idx];
                s.i = order[(order.indexOf(s.i) + 1) % order.length];
                this.writeString(false);
              }
              return true; // swallow either way -- it's our chip row
            }
          }

          // 2. existing handle -> start a drag
          let hitIdx = -1;
          let bestD = Infinity;
          for (let idx = 0; idx < this.stops.length; idx++) {
            const hp = this.stops[idx]._handlePos;
            if (!hp) continue;
            const d = Math.hypot(hp.x - px, hp.y - py);
            if (d <= HANDLE_HIT_R && d < bestD) {
              bestD = d;
              hitIdx = idx;
            }
          }
          if (hitIdx !== -1) {
            this.dragIndex = hitIdx;
            this.dragRemoving = false;
            this._dragLastPx = px;
            this._dragLastPy = py;
            // spec: bump _order the moment a stop is picked up, so a
            // freshly-dragged stop always sorts last among position ties.
            this.stops[hitIdx]._order = this.orderCounter++;
            node.setDirtyCanvas(true, true);
            return true;
          }

          // 3. double-click on the empty strip -> add a stop there
          const inStrip = px >= x0 && px <= x0 + w && py >= stripY && py <= stripY + STRIP_H;
          if (inStrip) {
            const now = performance.now();
            const last = this._lastClickPos;
            const isDbl =
              now - this._lastClickTime < DBLCLICK_MS &&
              last &&
              Math.hypot(last[0] - px, last[1] - py) < DBLCLICK_PX;
            this._lastClickTime = now;
            this._lastClickPos = [px, py];
            if (isDbl && this.stops.length < MAX_STOPS) {
              const p = clamp01((px - x0) / w);
              const v = clamp01(1 - (py - stripY) / STRIP_H);
              this.stops.push({ p, v, i: "linear", _order: this.orderCounter++ });
              this._lastClickTime = 0; // don't chain a 3rd click into another add
              this._lastClickPos = null;
              this.writeString(true);
            }
            return true; // swallow single clicks in the strip too (no node drag)
          }

          // 4. dead zone within the chip row but not on any chip -- still ours
          if (py >= chipY && py <= chipY + CHIP_H) return true;

          return false;
        }

        if (isMove && this.dragIndex !== -1) {
          const s = this.stops[this.dragIndex];
          if (event.shiftKey && this._dragLastPx != null) {
            const dp = ((px - this._dragLastPx) / w) * FINE_SCALE;
            const dv = (-(py - this._dragLastPy) / STRIP_H) * FINE_SCALE;
            s.p = clamp01(s.p + dp);
            s.v = clamp01(s.v + dv);
          } else {
            s.p = clamp01((px - x0) / w);
            s.v = clamp01(1 - (py - stripY) / STRIP_H);
          }
          this._dragLastPx = px;
          this._dragLastPy = py;

          const removing = py < stripY - REMOVE_THRESHOLD || py > stripY + STRIP_H + REMOVE_THRESHOLD;
          this.dragRemoving = removing && this.stops.length > 1;

          this.writeString(false); // live feedback into the STRING widget; index stays valid
          node.setDirtyCanvas(true, true);
          return true;
        }

        if (isUp && this.dragIndex !== -1) {
          const idx = this.dragIndex;
          if (this.dragRemoving && this.stops.length > 1) {
            this.stops.splice(idx, 1);
          } else {
            const dragged = this.stops[idx];
            for (let k = 0; k < this.stops.length; k++) {
              if (k === idx) continue;
              if (Math.abs(this.stops[k].p - dragged.p) < DUP_EPS_P) {
                // duplicate-position rule: the JUST-DRAGGED stop's _order was
                // already bumped to the max on pickup, so snapping p exactly
                // onto the other stop makes it sort after (later wins).
                dragged.p = this.stops[k].p;
                break;
              }
            }
          }
          this.dragIndex = -1;
          this.dragRemoving = false;
          this._dragLastPx = null;
          this._dragLastPy = null;
          this.writeString(true);
          return true;
        }

        return false;
      } catch (err) {
        console.error("[FieldGradient] ramp widget mouse() failed:", err);
        this.dragIndex = -1;
        this.dragRemoving = false;
        return false;
      }
    },
  };

  return widget;
}

// --- attach to the node ---------------------------------------------------

function attachRampWidget(node) {
  const stringWidget = node.widgets?.find((w) => w.name === "ramp");
  if (!stringWidget) {
    console.warn("[FieldGradient] ramp STRING widget not found; canvas widget not attached");
    return;
  }
  if (node._fieldGradientRampAttached) return;
  node._fieldGradientRampAttached = true;

  const widget = createRampWidget(node, stringWidget);
  node.addCustomWidget(widget);
  widget.reparse();

  // Place the canvas widget directly ABOVE the STRING widget (which stays
  // visible below it as the manual-JSON-paste escape hatch).
  const stringIdx0 = node.widgets.indexOf(stringWidget);
  const widgetIdx0 = node.widgets.indexOf(widget);
  if (widgetIdx0 !== -1 && stringIdx0 !== -1 && widgetIdx0 !== stringIdx0 - 1) {
    node.widgets.splice(widgetIdx0, 1);
    const stringIdx1 = node.widgets.indexOf(stringWidget);
    node.widgets.splice(stringIdx1, 0, widget);
  }

  // Re-sync the canvas whenever the STRING widget changes directly -- via
  // its callback (fires on committed edits) and its textarea's own `input`
  // event (fires live, for multiline widgets that expose .inputEl).
  const origCallback = stringWidget.callback;
  stringWidget.callback = function (value, ...rest) {
    const out = origCallback ? origCallback.apply(this, [value, ...rest]) : undefined;
    widget.reparse();
    node.setDirtyCanvas(true, true);
    return out;
  };

  const hookInputEl = () => {
    if (stringWidget.inputEl && !stringWidget.inputEl._fieldGradientHooked) {
      stringWidget.inputEl._fieldGradientHooked = true;
      stringWidget.inputEl.addEventListener("input", () => {
        widget.reparse();
        node.setDirtyCanvas(true, true);
      });
    }
  };
  hookInputEl();
  // Multiline STRING widgets sometimes create their <textarea> a tick after
  // onNodeCreated returns -- retry once on the next frame.
  setTimeout(hookInputEl, 0);

  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "AKURATE.FieldGradientRamp",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== "FieldGradient") return;
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
      attachRampWidget(this);
      return r;
    };
  },
});
