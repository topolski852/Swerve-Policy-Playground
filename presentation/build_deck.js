// build_deck.js — "Teaching a Robot to Drive Itself" (RL intro, Advanced Programming lesson 1)
//
//   python presentation/prep_media.py     # copy clips + posters into presentation/media
//   node presentation/build_deck.js       # writes presentation/RL_Intro_Lesson1.pptx
//
// prep_media.py also writes media/charts.json (Chapter 3 charts) from the training logs.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const fa = require("react-icons/fa");
const { applyTheme } = require(path.join(__dirname, "apply_theme.js"));

const HERE = __dirname;
const MEDIA = path.join(HERE, "media");
const OUT = path.join(HERE, "RL_Intro_Lesson1.pptx");
const manifest = JSON.parse(fs.readFileSync(path.join(MEDIA, "manifest.json"), "utf8"));
const charts = JSON.parse(fs.readFileSync(path.join(MEDIA, "charts.json"), "utf8"));

// ── Theme ─────────────────────────────────────────────────────────────────────
// Dark "night field" ground so the sim videos sit naturally; yellow = reward
// (the target-waypoint dot in every video), red/blue = alliances, green = module arrows.
const THEME = {
  name: "Robot Learning",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14171C", lt1: "FFFFFF", dk2: "232A35", lt2: "C9D1DC",
    accent1: "F5C518", // reward yellow
    accent2: "E5484D", // red alliance / penalty
    accent3: "3B82F6", // blue alliance
    accent4: "3DDC84", // module-arrow green / good
    accent5: "8A94A6", // muted
    accent6: "FF8A3D",
    hlink: "F5C518", folHlink: "C9D1DC",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Teaching a Robot to Drive Itself";
pres.author = "Team 1507 Warlocks";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, H = 7.5, M = 0.6;

// ── Layouts ───────────────────────────────────────────────────────────────────
pres.defineSlideMaster({
  title: "COVER",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: 6.2, h: 2.0,
        fontSize: 48, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: 6.0, h: 1.0,
        fontSize: 22, color: C.accent1, valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "SECTION",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 1.9, w: 8, h: 0.5,
        fontSize: 18, bold: true, color: C.accent1, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.45, w: 11.5, h: 1.3,
        fontSize: 44, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.9, w: 7.6, h: 1.6,
        fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 0.85,
        fontSize: 34, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
});

// ── Helpers ───────────────────────────────────────────────────────────────────
async function icon(Comp, hex, px = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: "#" + hex, size: px }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function coverPng(jpg) {
  const buf = await sharp(jpg).png().toBuffer();
  return "data:image/png;base64," + buf.toString("base64");
}

function shadow() {
  return { type: "outer", color: "000000", blur: 8, offset: 3, angle: 90, opacity: 0.35 };
}

// Icon inside a filled circle (the deck's recurring motif).
async function badge(slide, Comp, x, y, d, fill, iconHex, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" },
    objectName: name + " circle" });
  const s = d * 0.52;
  slide.addImage({ data: await icon(Comp, iconHex), x: x + (d - s) / 2, y: y + (d - s) / 2, w: s, h: s,
    objectName: name + " icon", altText: name });
}

function card(slide, x, y, w, h, name, fill = C.text2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill }, line: { type: "none" }, shadow: shadow(), objectName: name });
}

async function video(slide, key, x, y, w, name) {
  const m = manifest[key];
  const h = w * (m.h / m.w);
  // thin frame so dark video edges don't melt into the background
  slide.addShape(pres.shapes.RECTANGLE, { x: x - 0.04, y: y - 0.04, w: w + 0.08, h: h + 0.08,
    fill: { color: HEX.dk2 }, line: { color: HEX.accent5, width: 0.75 }, objectName: name + " frame" });
  slide.addMedia({ type: "video", path: m.mp4, cover: await coverPng(m.poster), x, y, w, h,
    objectName: name });
  return h;
}

function placeholderVideo(slide, x, y, w, h, msg, name) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: HEX.dk2 },
    line: { color: HEX.accent5, width: 1, dashType: "dash" }, objectName: name });
  slide.addText(msg, { x, y, w, h, isTextBox: true, align: "center", valign: "middle",
    fontSize: 16, color: C.accent5, italic: true });
}

// "Score" label: green for positive, red for negative
function scoreRun(v) {
  return { text: (v > 0 ? "+" : "−") + Math.abs(v).toFixed(0),
    options: { bold: true, color: v > 0 ? C.accent4 : C.accent2 } };
}

function caption(slide, x, y, w, runs, name, h = 0.9) {
  slide.addText(runs, { x, y, w, h, isTextBox: true, fontSize: 14, color: C.background2,
    valign: "top", margin: 0, objectName: name, paraSpaceAfter: 2 });
}

const steps = (n) => n.toLocaleString("en-US");

// ── Observation chips ─────────────────────────────────────────────────────────
// One rounded chip per number the robot observes, colored by group, so students
// can literally count the robot's inputs.
const OBS_GROUPS = {
  speed:    { fill: HEX.accent3, txt: HEX.lt1, name: "Speed" },
  waypoint: { fill: HEX.accent1, txt: HEX.dk1, name: "Next 2 waypoints" },
  path:     { fill: HEX.accent6, txt: HEX.dk1, name: "Place on the path" },
  heading:  { fill: HEX.accent5, txt: HEX.dk1, name: "Facing" },
  position: { fill: HEX.accent4, txt: HEX.dk1, name: "Place on the field" },
  // violet, not accent6: orange already means "Place on the path" in version 1's chips
  rays:     { fill: "A78BFA", txt: HEX.dk1, name: "Distance rays" },
};
const OBS_V1 = [
  ["Speed X", "speed"], ["Speed Y", "speed"],
  ["Next X", "waypoint"], ["Next Y", "waypoint"], ["After X", "waypoint"], ["After Y", "waypoint"],
  ["Progress", "path"], ["Off line", "path"], ["Heading", "heading"],
];
const OBS_NOW = [
  ["Speed X", "speed"], ["Speed Y", "speed"],
  ["Field X", "position"], ["Field Y", "position"],
  ["Next X", "waypoint"], ["Next Y", "waypoint"], ["After X", "waypoint"], ["After Y", "waypoint"],
];
// 8 rays, 45° apart, field-relative (path_randomizer/rays.py: ray 0 points +x, then counter-clockwise)
const OBS_RAYS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"].map((d) => ["Ray " + d, "rays"]);

// opts.dropped: Set of indexes to grey out + strike; opts.isNew: Set of indexes to tag NEW
function chipRow(slide, chips, x, y, cw, ch, gap, name, opts = {}) {
  chips.forEach(([label, g], i) => {
    const cx = x + i * (cw + gap), grp = OBS_GROUPS[g];
    const dropped = opts.dropped && opts.dropped.has(i);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y, w: cw, h: ch, rectRadius: 0.1,
      fill: { color: dropped ? HEX.dk2 : grp.fill },
      line: dropped ? { color: HEX.accent5, width: 1, dashType: "dash" } : { type: "none" },
      objectName: `${name} chip ${i + 1}` });
    slide.addText(label, { x: cx, y, w: cw, h: ch, isTextBox: true, align: "center", valign: "middle",
      fontSize: 14, bold: true, margin: 0, color: dropped ? HEX.accent5 : grp.txt, strike: dropped ? "sngStrike" : undefined });
    if (opts.isNew && opts.isNew.has(i)) {
      slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx + cw / 2 - 0.3, y: y - 0.2, w: 0.6, h: 0.3, rectRadius: 0.08,
        fill: { color: HEX.lt1 }, line: { type: "none" }, objectName: `${name} new tag ${i + 1}` });
      slide.addText("NEW", { x: cx + cw / 2 - 0.3, y: y - 0.2, w: 0.6, h: 0.3, isTextBox: true, align: "center",
        valign: "middle", fontSize: 11, bold: true, color: HEX.dk1, margin: 0 });
    }
  });
}

// ── Slides ────────────────────────────────────────────────────────────────────
async function build() {
  let s;

  // 1 ─ Cover
  pres.addSection({ title: "Intro" });
  s = pres.addSlide({ masterName: "COVER", sectionTitle: "Intro" });
  s.addText("Teaching a Robot to Drive Itself", { placeholder: "title" });
  s.addText("A friendly intro to Reinforcement Learning (RL)", { placeholder: "body" });
  s.addText("Team 1507 Warlocks  ·  Advanced Programming  ·  Lesson 1", {
    x: M, y: 6.5, w: 7, h: 0.4, isTextBox: true, fontSize: 14, color: C.accent5, margin: 0 });
  {
    const m = manifest.hook, w = 5.9, h = w * m.h / m.w;
    s.addShape(pres.shapes.RECTANGLE, { x: 7.0 - 0.04, y: (H - h) / 2 - 0.04, w: w + 0.08, h: h + 0.08,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent5, width: 0.75 }, objectName: "cover frame" });
    s.addImage({ path: m.poster, x: 7.0, y: (H - h) / 2, w, h, objectName: "cover image",
      altText: "Simulated robot driving a figure-8 on the FRC field" });
  }
  s.addNotes("Welcome to our first advanced programming lesson. Today: how we taught a simulated robot to drive on its own, without writing any steering code. Lots of videos, very little math.");

  // 2 ─ Hook video
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Intro" });
  s.addText("Nobody wrote code to steer this path", { placeholder: "title" });
  {
    const w = 10.0, x = (W - w) / 2;
    const h = await video(s, "hook", x, 1.5, w, "hook video");
    caption(s, x, 1.5 + h + 0.25, w, [
      { text: "A full figure-8 around both hubs in about 5 seconds, with no crashes. ", options: { bold: true, color: C.background1 } },
      { text: `The robot taught itself to do this after ${steps(manifest.hook.train_step)} practice steps (about 17 minutes of simulated driving).` },
    ], "hook caption", 1.0);
  }
  s.addNotes("Play the video. Ask: how do you think we programmed this? The answer: we didn't program the steering at all. We only told it what counts as 'good' and 'bad', and it figured out the rest. 50 steps = 1 simulated second, so 50,000 steps is about 17 minutes of practice.");

  // 3 ─ Two ways to learn
  pres.addSection({ title: "What is RL?" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What is RL?" });
  s.addText("Two ways a computer can learn", { placeholder: "title" });
  s.addText([
    { text: "Both are kinds of " },
    { text: "Machine Learning (ML)", options: { bold: true, color: C.background1 } },
    { text: ", which is part of " },
    { text: "Artificial Intelligence (AI)", options: { bold: true, color: C.background1 } },
    { text: ". The computer gets better from experience instead of following rules we typed in." },
  ], { x: M, y: 1.35, w: W - 2 * M, h: 0.8, isTextBox: true, fontSize: 16, color: C.background2, margin: 0 });
  {
    const cw = 5.85, ch = 3.9, y = 2.4;
    const cols = [
      { x: M, Comp: fa.FaTags, fill: HEX.accent3, head: "Supervised Learning",
        line: "Learn from examples that come with the right answers.",
        like: "Like studying with an answer key." },
      { x: W - M - cw, Comp: fa.FaStar, fill: HEX.accent1, head: "Reinforcement Learning (RL)",
        line: "Try things, get points for good results, lose points for bad ones.",
        like: "Like getting better at a video game by chasing a high score." },
    ];
    for (const c of cols) {
      card(s, c.x, y, cw, ch, c.head + " card");
      await badge(s, c.Comp, c.x + 0.4, y + 0.4, 1.0, c.fill, HEX.dk1, c.head);
      s.addText(c.head, { x: c.x + 0.4, y: y + 1.6, w: cw - 0.8, h: 0.6, isTextBox: true,
        fontSize: 24, bold: true, color: C.background1, margin: 0 });
      s.addText([
        { text: c.line, options: { breakLine: true } },
        { text: c.like, options: { italic: true, color: c.fill === HEX.accent1 ? C.accent1 : C.background2 } },
      ], { x: c.x + 0.4, y: y + 2.25, w: cw - 0.8, h: 1.4, isTextBox: true, fontSize: 18,
        color: C.background2, margin: 0, valign: "top", paraSpaceAfter: 8 });
    }
  }
  s.addNotes("Two big families of machine learning. Supervised: you show it lots of examples with the correct answer. Reinforcement learning: nobody gives it the answer; it gets a score and has to figure out how to raise it.");

  // 4 ─ Last season's shooter
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What is RL?" });
  s.addText("Last season: our shooter learned from an answer key", { placeholder: "title" });
  {
    const stepsArr = [
      { Comp: fa.FaBasketballBall, head: "1. Shoot", body: "The robot records its distance to the hub and its flywheel speed in RPM (revolutions per minute)." },
      { Comp: fa.FaClipboardCheck, head: "2. Label", body: "A person marks every shot: made or missed." },
      { Comp: fa.FaChartLine, head: "3. Fit a curve", body: "Keep only the made shots and find the best curve: speed = a·d + b·d² + c" },
      { Comp: fa.FaBullseye, head: "4. Use it", body: "For any distance, the robot looks up the speed that worked." },
    ];
    const cw = 2.8, gap = 0.3, y = 1.65, ch = 3.4;
    for (let i = 0; i < stepsArr.length; i++) {
      const st = stepsArr[i], x = M + i * (cw + gap);
      card(s, x, y, cw, ch, st.head + " card");
      await badge(s, st.Comp, x + 0.3, y + 0.3, 0.8, HEX.accent3, HEX.lt1, st.head);
      s.addText(st.head, { x: x + 0.3, y: y + 1.25, w: cw - 0.6, h: 0.45, isTextBox: true,
        fontSize: 20, bold: true, color: C.background1, margin: 0 });
      s.addText(st.body, { x: x + 0.3, y: y + 1.75, w: cw - 0.6, h: 1.5, isTextBox: true,
        fontSize: 15, color: C.background2, margin: 0, valign: "top" });
      if (i < stepsArr.length - 1) {
        s.addShape(pres.shapes.LINE, { x: x + cw + 0.04, y: y + ch / 2, w: gap - 0.08, h: 0,
          line: { color: HEX.accent5, width: 2, endArrowType: "triangle" }, objectName: "flow arrow " + (i + 1) });
      }
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.4, w: W - 2 * M, h: 1.15, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent3, width: 1.5 }, objectName: "takeaway outline" });
    s.addText([
      { text: "We were the teacher with the answer key. ", options: { bold: true, color: C.background1 } },
      { text: "That's supervised learning. The robot never tried anything on its own; it copied what worked." },
    ], { x: M + 0.35, y: 5.4, w: W - 2 * M - 0.7, h: 1.15, isTextBox: true, fontSize: 17,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("Remember the shooter from last year? The robot logged each shot, we told it which went in, and the code fit a curve through the good shots. Useful, but we did all the judging. This year is different.");

  // 5 ─ Dog training analogy
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What is RL?" });
  s.addText("This season: the robot learns from points", { placeholder: "title" });
  {
    const cw = 5.85, ch = 4.3, y = 1.6;
    const cols = [
      { x: M, Comp: fa.FaDog, fill: HEX.accent6, head: "Training a dog",
        good: "Sits when asked → treat", bad: "Jumps on the couch → no treat",
        punch: "Nobody explains HOW to sit. The dog figures it out." },
      { x: W - M - cw, Comp: fa.FaRobot, fill: HEX.accent4, head: "Training our robot",
        good: "Reaches the next waypoint → +points", bad: "Crashes into a hub → −points",
        punch: "Nobody explains HOW to steer. The robot figures it out." },
    ];
    for (const c of cols) {
      card(s, c.x, y, cw, ch, c.head + " card");
      await badge(s, c.Comp, c.x + 0.4, y + 0.4, 1.1, c.fill, HEX.dk1, c.head);
      s.addText(c.head, { x: c.x + 1.75, y: y + 0.55, w: cw - 2.1, h: 0.8, isTextBox: true,
        fontSize: 26, bold: true, color: C.background1, margin: 0, valign: "middle" });
      s.addText([
        { text: "✔  " + c.good, options: { color: C.accent4, breakLine: true } },
        { text: "✖  " + c.bad, options: { color: C.accent2 } },
      ], { x: c.x + 0.4, y: y + 1.8, w: cw - 0.8, h: 1.0, isTextBox: true, fontSize: 19,
        margin: 0, paraSpaceAfter: 8 });
      s.addText(c.punch, { x: c.x + 0.4, y: y + 3.0, w: cw - 0.8, h: 1.0, isTextBox: true,
        fontSize: 18, italic: true, color: C.accent1, margin: 0, valign: "top" });
    }
  }
  s.addNotes("The easiest way to picture reinforcement learning: training a dog. You never explain the muscle movements for 'sit'. You reward the result. Same with our robot: we reward results, and the robot works out the steering.");

  // 6 ─ The RL loop
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What is RL?" });
  s.addText("The learning loop", { placeholder: "title" });
  {
    // four boxes around a cycle
    const bw = 2.9, bh = 1.45, off = 1.0; // off = how far side boxes sit from the center line
    const cx = M + off + bw, cy = 4.15;   // cycle center; left box starts at the margin
    const nodes = [
      { key: "Policy", sub: "the robot's brain", x: cx - bw / 2, y: 1.45, fill: HEX.accent1, txt: HEX.dk1, Comp: fa.FaBrain },
      { key: "Action", sub: "drive speed in X and Y", x: cx + off, y: cy - bh / 2, fill: HEX.accent3, txt: HEX.lt1, Comp: fa.FaGamepad },
      { key: "Environment", sub: "the simulated field", x: cx - bw / 2, y: 6.85 - bh, fill: HEX.accent4, txt: HEX.dk1, Comp: fa.FaGlobeAmericas },
      { key: "Observation + Reward", sub: "what it sees + its score", x: cx - off - bw, y: cy - bh / 2, fill: HEX.accent6, txt: HEX.dk1, Comp: fa.FaEye },
    ];
    // arrows (drawn first so boxes sit on top)
    const arrow = (x1, y1, x2, y2, n) => s.addShape(pres.shapes.LINE, {
      x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
      flipH: x2 < x1, flipV: y2 < y1,
      line: { color: HEX.lt2, width: 2.5, endArrowType: "triangle" }, objectName: "loop arrow " + n });
    const P = nodes[0], A = nodes[1], E = nodes[2], O = nodes[3];
    arrow(P.x + bw, P.y + bh / 2, A.x + bw / 2, A.y, 1);
    arrow(A.x + bw / 2, A.y + bh, E.x + bw, E.y + bh / 2, 2);
    arrow(E.x, E.y + bh / 2, O.x + bw / 2, O.y + bh, 3);
    arrow(O.x + bw / 2, O.y, P.x, P.y + bh / 2, 4);
    for (const n of nodes) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: n.x, y: n.y, w: bw, h: bh, rectRadius: 0.15,
        fill: { color: n.fill }, line: { type: "none" }, shadow: shadow(), objectName: n.key + " box" });
      s.addImage({ data: await icon(n.Comp, n.txt), x: n.x + 0.25, y: n.y + (bh - 0.6) / 2, w: 0.6, h: 0.6,
        objectName: n.key + " icon", altText: n.key });
      s.addText([
        { text: n.key, options: { bold: true, fontSize: 19, breakLine: true } },
        { text: n.sub, options: { fontSize: 14 } },
      ], { x: n.x + 1.0, y: n.y, w: bw - 1.1, h: bh, isTextBox: true, color: n.txt, valign: "middle", margin: 0 });
    }
    // side explainer
    const sx = 9.0, sw = W - M - sx;
    card(s, sx, 1.45, sw, 5.4, "loop explainer card");
    s.addText([
      { text: "Around and around", options: { bold: true, fontSize: 20, color: C.background1, breakLine: true } },
      { text: "1.  The policy looks at what the robot sees.", options: { breakLine: true } },
      { text: "2.  It picks an action: how fast to drive.", options: { breakLine: true } },
      { text: "3.  The simulator moves the robot.", options: { breakLine: true } },
      { text: "4.  The robot gets a reward (its score for that moment).", options: { breakLine: true } },
      { text: "This happens 50 times per simulated second. After thousands of loops, the policy learns which actions earn the most points.",
        options: { color: C.accent1, italic: true } },
    ], { x: sx + 0.3, y: 1.7, w: sw - 0.6, h: 5.0, isTextBox: true, fontSize: 15, color: C.background2,
      valign: "top", margin: 0, paraSpaceAfter: 10 });
  }
  s.addNotes("Walk around the loop. The policy is a small neural network. Every 20 milliseconds (the same loop timing as our real robot code) it sees, acts, and gets scored. Training means nudging the network so actions that led to more points become more likely. The method we use is called SAC (Soft Actor-Critic). It trains two networks: the actor is the policy that drives, and the critic is a second network that predicts how many points are coming. That critic shows up again in Chapter 3, when a bug fooled it.");

  // 7 ─ Words to know
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What is RL?" });
  s.addText("Words to know", { placeholder: "title" });
  {
    const words = [
      { w: "Policy", d: "The robot's brain: a small neural network that turns what it sees into what it does.", Comp: fa.FaBrain, fill: HEX.accent1, ic: HEX.dk1 },
      { w: "Observation", d: "OBS for short. The numbers the robot “sees”, like its speed and where its next waypoint is.", Comp: fa.FaEye, fill: HEX.accent6, ic: HEX.dk1 },
      { w: "Action", d: "What the robot does: how fast to drive forward/back and left/right.", Comp: fa.FaGamepad, fill: HEX.accent3, ic: HEX.lt1 },
      { w: "Reward", d: "The score for each moment. Positive = good, negative = bad.", Comp: fa.FaStar, fill: HEX.accent1, ic: HEX.dk1 },
      { w: "Episode", d: "One attempt, from start until it finishes, crashes, or runs out of time.", Comp: fa.FaFlagCheckered, fill: HEX.accent4, ic: HEX.dk1 },
      { w: "Training step", d: "One tick of practice. 50 steps = 1 second of driving.", Comp: fa.FaStopwatch, fill: HEX.accent5, ic: HEX.dk1 },
    ];
    const cols = 3, cw = 3.85, ch = 2.45, gx = 0.3, gy = 0.3, y0 = 1.55;
    for (let i = 0; i < words.length; i++) {
      const it = words[i], x = M + (i % cols) * (cw + gx), y = y0 + Math.floor(i / cols) * (ch + gy);
      card(s, x, y, cw, ch, it.w + " card");
      await badge(s, it.Comp, x + 0.3, y + 0.3, 0.75, it.fill, it.ic, it.w);
      s.addText(it.w, { x: x + 1.25, y: y + 0.3, w: cw - 1.5, h: 0.75, isTextBox: true, fontSize: 21,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
      s.addText(it.d, { x: x + 0.3, y: y + 1.2, w: cw - 0.6, h: 1.15, isTextBox: true, fontSize: 15,
        color: C.background2, valign: "top", margin: 0 });
    }
  }
  s.addNotes("These six words come up in every slide from here on. Leave this up for a minute.");

  // 8 ─ Chapter 1 divider
  pres.addSection({ title: "Chapter 1: Empty field" });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Chapter 1: Empty field" });
  s.addText("CHAPTER 1", { placeholder: "kicker" });
  s.addText("An empty field", { placeholder: "title" });
  s.addText("Just a path and a robot: no walls, no hubs, nothing to hit. Can it learn to follow the line at all?", { placeholder: "body" });
  s.addNotes("We started as simple as possible. If it can't do this, nothing harder will work.");

  // 9 ─ Chapter 1 videos
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 1: Empty field" });
  s.addText("Wander, figure it out, optimize", { placeholder: "title" });
  {
    const clips = [
      { key: "ch1_500", head: "1. Wander", steps: "500 steps", score: -202,
        txt: "No idea what earns points yet. It drifts off and the attempt ends." },
      { key: "ch1_8k", head: "2. Figure it out", steps: "8,000 steps", score: -297,
        txt: "Reaches 9 of 14 waypoints, but pauses at each one: is the next one worth it?" },
      { key: "ch1_12k", head: "3. Optimize", steps: "12,000 steps", score: 271,
        txt: "Knows the whole path. Now it gets faster: a full loop in about 6 seconds." },
    ];
    const vw = 3.85, gap = 0.3, y = 1.6;
    for (let i = 0; i < clips.length; i++) {
      const c = clips[i], x = M + i * (vw + gap);
      const h = await video(s, c.key, x, y, vw, c.head + " video");
      s.addText(c.head, { x, y: y + h + 0.25, w: vw, h: 0.45, isTextBox: true, fontSize: 22, bold: true,
        color: C.accent1, margin: 0 });
      caption(s, x, y + h + 0.75, vw, [
        { text: c.steps + "  ·  Score: ", options: { bold: true, color: C.background1 } }, scoreRun(c.score),
        { text: "", options: { breakLine: true } },
        { text: c.txt },
      ], c.head + " caption", 1.0);
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.55, w: W - 2 * M, h: 1.1, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.5 }, objectName: "takeaway outline" });
    s.addText([
      { text: "Standing still costs points every second. ", options: { bold: true, color: C.background1 } },
      { text: "Each time it pauses at a waypoint, that cost nudges it on to the next one, until it learns the whole path." },
    ], { x: M + 0.35, y: 5.55, w: W - 2 * M - 0.7, h: 1.1, isTextBox: true, fontSize: 17,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("Play them left to right. Stage 1, wander: early on it is trying random moves on purpose, exploring to find out what earns points. Stage 2, figure it out: it has learned that waypoints are good and drives to them, but after each one it hesitates, unsure whether moving on is worth it or whether it should stay put. Staying still keeps costing points, so it keeps getting pushed toward the next waypoint. It reached 9 of 14 before time ran out. Fun question for students: why is the 8,000-step score (−297) LOWER than the 500-step score (−202) even though it got much further? Because it survived the whole 30 seconds paying the 'every second' cost, while the 500-step attempt ended early. Stage 3, optimize: it knows the whole path, and from here on training just makes it faster and smoother.");

  // 9b ─ Version 1 observations
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 1: Empty field" });
  s.addText("What the robot sees (version 1)", { placeholder: "title" });
  {
    s.addText("50 times a second, the policy gets these 9 numbers. That's ALL it knows about the world.", {
      x: M, y: 1.35, w: W - 2 * M, h: 0.5, isTextBox: true, fontSize: 18, color: C.background2, margin: 0 });
    const cw = 1.22, gap = 0.14, ch = 1.0, y = 2.3;
    const x0 = (W - (OBS_V1.length * cw + (OBS_V1.length - 1) * gap)) / 2;
    chipRow(s, OBS_V1, x0, y, cw, ch, gap, "version 1");
    // group cards under the chips they describe
    const groups = [
      { g: "speed", from: 0, to: 1, q: "How fast am I going, forward/back and left/right?" },
      { g: "waypoint", from: 2, to: 5, q: "Which way, and how far, to my next waypoint and the one after it?" },
      { g: "path", from: 6, to: 7, q: "How far around the loop am I? How far off the line?" },
      { g: "heading", from: 8, to: 8, q: "Which way am I pointing?" },
    ];
    for (const gr of groups) {
      const gx = x0 + gr.from * (cw + gap), gw = (gr.to - gr.from + 1) * cw + (gr.to - gr.from) * gap;
      const grp = OBS_GROUPS[gr.g];
      s.addShape(pres.shapes.LINE, { x: gx, y: y + ch + 0.15, w: gw, h: 0,
        line: { color: grp.fill, width: 3 }, objectName: grp.name + " bracket" });
      s.addText([
        { text: grp.name, options: { bold: true, color: grp.fill, breakLine: true } },
        { text: gr.q },
      ], { x: gx, y: y + ch + 0.35, w: gw, h: 1.6, isTextBox: true, fontSize: 16, color: C.background2,
        valign: "top", margin: 0, paraSpaceAfter: 4 });
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.3, w: W - 2 * M, h: 1.1, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.5 }, objectName: "takeaway outline" });
    s.addText([
      { text: "Notice what's missing: where it is on the field. ", options: { bold: true, color: C.background1 } },
      { text: "It only knows where it is compared to the path. Remember that for Chapter 3!" },
    ], { x: M + 0.35, y: 5.3, w: W - 2 * M - 0.7, h: 1.1, isTextBox: true, fontSize: 17,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("These are the observations (OBS) for the first figure-8 experiment, and they stayed exactly the same for all four of the figure-8 recordings, even after we added the real field. Each chip is one number. The waypoint numbers are measured from the robot's own point of view. Ask: if you could only see these 9 numbers, could you drive the path? Notice there is no field position: it knows how far around the loop it is and how far off the line, but not where on the field it is.");

  // 10 ─ Chapter 1 chart + how the score worked
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 1: Empty field" });
  s.addText("The “aha!” moment", { placeholder: "title" });
  {
    const labels = ["500", "1k", "2k", "5k", "8k", "12k", "18k", "25k", "35k", "50k", "75k", "100k", "150k", "200k", "300k", "500k"];
    const vals = [-202, -202, -202, -202, -297, 271, 274, 284, 286, 289, 290, 291, 292, 290, 292, 294];
    card(s, M, 1.45, 7.4, 5.35, "chart card");
    s.addChart(pres.charts.LINE, [{ name: "Test-drive score", labels, values: vals }], {
      x: M + 0.2, y: 1.6, w: 7.0, h: 5.05,
      showTitle: true, title: "Test-drive score vs. practice steps", titleColor: HEX.lt1, titleFontSize: 15,
      titleFontFace: "+mn-lt",
      chartColors: [HEX.accent1], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 7,
      catAxisLabelColor: HEX.lt2, valAxisLabelColor: HEX.lt2, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "3A4350", size: 0.5 }, catGridLine: { style: "none" },
      catAxisLineShow: false, valAxisLineShow: false,
      showCatAxisTitle: true, catAxisTitle: "Practice steps", catAxisTitleColor: HEX.lt2, catAxisTitleFontSize: 12,
      showValAxisTitle: true, valAxisTitle: "Score", valAxisTitleColor: HEX.lt2, valAxisTitleFontSize: 12,
      showLegend: false, valAxisMinVal: -400, valAxisMaxVal: 400, valAxisMajorUnit: 200,
      plotArea: { fill: { color: HEX.dk2 } }, objectName: "chapter 1 score chart",
    });
    const rx = M + 7.7, rw = W - M - rx;
    s.addText("How it earned points", { x: rx, y: 1.45, w: rw, h: 0.5, isTextBox: true, fontSize: 20, bold: true,
      color: C.background1, margin: 0 });
    // Reward weights as they were for the first recorded run (commit 2262adf).
    const plus = [
      ["+1", "per meter forward along the path"],
      ["+0.8", "for heading toward the next waypoint"],
      ["+2", "for each waypoint passed"],
      ["+20", "for finishing the loop"],
      ["−0.6", "per meter off the line"],
      ["−0.3", "for jerky back-and-forth wiggles"],
      ["−0.02", "every step (hurry up!)"],
      ["−10", "and it's over if it gets 2.5 m off the path"],
    ];
    s.addText(plus.flatMap(([pts, t], i) => [
      { text: pts + "  ", options: { bold: true, color: pts.startsWith("+") ? C.accent4 : C.accent2 } },
      { text: t, options: { color: C.background2, breakLine: i < plus.length - 1 } },
    ]), { x: rx, y: 2.0, w: rw, h: 3.4, isTextBox: true, fontSize: 15, margin: 0, valign: "top", paraSpaceAfter: 5 });
    s.addText("After 12,000 steps the score barely moves. It's already about as good as these points can make it.", {
      x: rx, y: 5.55, w: rw, h: 1.1, isTextBox: true, fontSize: 15, italic: true, color: C.accent1, margin: 0, valign: "top" });
  }
  s.addNotes("Each dot is the score from one test drive recorded during training. Flat and negative while it explores, then a jump between 8,000 and 12,000 steps. The right side is the whole reward function for this run, with the real numbers: that list is all the robot was told. Points are added up every step (50 times a second), so small numbers add up fast: −0.02 per step is −1 per second.");

  // 11 ─ Chapter 2 divider
  pres.addSection({ title: "Chapter 2: Real field" });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Chapter 2: Real field" });
  s.addText("CHAPTER 2", { placeholder: "kicker" });
  s.addText("The real FRC field", { placeholder: "title" });
  s.addText("The FIRST Robotics Competition (FRC) 2026 field, our real robot's size, and a figure-8 around both hubs. New rule: hitting a hub, trench, or wall costs points and ends the attempt.", { placeholder: "body" });
  s.addNotes("Now we make it realistic. Obstacles are the red and blue shaded zones in the videos.");

  // 12 ─ Obstacles make it harder
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 2: Real field" });
  s.addText("Obstacles make it much harder", { placeholder: "title" });
  {
    const clips = [
      { key: "ch2_25k", head: "25,000 steps", score: -75, txt: "Still stuck. Runs out the 30-second clock." },
      { key: "ch2_100k", head: "100,000 steps", score: 217, txt: "Finally! The full figure-8 in about 5 seconds." },
    ];
    const vw = 5.9, gap = W - 2 * M - 2 * vw, y = 1.55;
    for (let i = 0; i < clips.length; i++) {
      const c = clips[i], x = M + i * (vw + gap);
      const h = await video(s, c.key, x, y, vw, c.head + " video");
      s.addText(c.head, { x, y: y + h + 0.25, w: vw, h: 0.45, isTextBox: true, fontSize: 22, bold: true,
        color: C.accent1, margin: 0 });
      caption(s, x, y + h + 0.75, vw, [{ text: "Score: " }, scoreRun(c.score), { text: "   " + c.txt }],
        c.head + " caption", 0.5);
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.75, w: W - 2 * M, h: 0.95, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.5 }, objectName: "takeaway outline" });
    s.addText([
      { text: "It never SAW the obstacles. ", options: { bold: true, color: C.background1 } },
      { text: "Same 9 numbers as before. It only lost 25 points when it hit one, so it learned to stay on a path we drew around them. That took about 6× more practice (75,000 steps)." },
    ], { x: M + 0.35, y: 5.75, w: W - 2 * M - 0.7, h: 0.95, isTextBox: true, fontSize: 17,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("Harder world = more practice. The first clean lap showed up at 75,000 steps (the 75k test drive), versus 12,000 on the empty field.");

  // 13 ─ Reward hacking
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 2: Real field" });
  s.addText("The robot outsmarted us", { placeholder: "title" });
  {
    const tricks = [
      { Comp: fa.FaSyncAlt, head: "Circle farming",
        t: "It circled waypoint 1 over and over to grab the bonus again and again.",
        fix: "Fix: only reward heading toward the NEXT waypoint, plus a small cost every second." },
      { Comp: fa.FaUndo, head: "Backwards is forwards",
        t: "The path is a loop, so backing up at the start fooled the scorekeeper into thinking it drove a whole lap.",
        fix: "Fix: only measure progress near where the robot really is." },
      { Comp: fa.FaParking, head: "Just… stop",
        t: "It learned that parking near the hub was “safer” than risking a crash penalty.",
        fix: "Fix: make finishing worth way more (next slide)." },
    ];
    const lw = 6.55, ch = 1.6, gy = 0.2, y0 = 1.45;
    for (let i = 0; i < tricks.length; i++) {
      const t = tricks[i], y = y0 + i * (ch + gy);
      card(s, M, y, lw, ch, t.head + " card");
      await badge(s, t.Comp, M + 0.25, y + 0.25, 0.7, HEX.accent2, HEX.lt1, t.head);
      s.addText([
        { text: t.head, options: { bold: true, fontSize: 17, color: C.background1, breakLine: true } },
        { text: t.t, options: { fontSize: 14, breakLine: true } },
        { text: t.fix, options: { fontSize: 14, color: C.accent4 } },
      ], { x: M + 1.15, y: y + 0.1, w: lw - 1.35, h: ch - 0.2, isTextBox: true, color: C.background2,
        valign: "middle", margin: 0, paraSpaceAfter: 3 });
    }
    const vx = M + lw + 0.35, vw = W - M - vx;
    const h = await video(s, "ch2_park", vx, 1.45, vw, "parking video");
    caption(s, vx, 1.45 + h + 0.2, vw, [
      { text: "150,000 steps: ", options: { bold: true, color: C.accent1 } },
      { text: "drives half the figure-8, then parks for the last 25 seconds." },
    ], "parking caption", 0.8);
    s.addText("The robot does exactly what the points say, not what we meant.", {
      x: vx, y: 1.45 + h + 1.15, w: vw, h: 1.2, isTextBox: true, fontSize: 20, bold: true, italic: true,
      color: C.accent1, margin: 0, valign: "top" });
  }
  s.addNotes("This is called 'reward hacking' and it happens to professional AI researchers too. Each time, the robot found a loophole in our scoring. Ask the students: how would YOU have fixed the parking one?");

  // 14 ─ Turning the knobs
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 2: Real field" });
  s.addText("Turning the knobs", { placeholder: "title" });
  {
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.dk1, fill: { color: HEX.accent1 }, fontSize: 14 } });
    const cell = (t, o = {}) => ({ text: t, options: { color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 15, ...o } });
    const rows = [
      [hdr("Version"), hdr("Waypoint bonus"), hdr("Finish bonus"), hdr("First full figure-8")],
      [cell("Version 1"), cell("+2"), cell("+20"), cell("75,000 steps")],
      [cell("Version 2 (extra waypoints)"), cell("+4"), cell("+20"), cell("Never: it parked", { color: HEX.accent2, bold: true })],
      [cell("Version 3"), cell("+6"), cell("+80", { bold: true, color: HEX.accent1 }), cell("18,000 steps", { color: HEX.accent4, bold: true })],
    ];
    s.addTable(rows, { x: M, y: 1.5, w: 6.9, colW: [2.4, 1.45, 1.35, 1.7], rowH: 0.6,
      border: { type: "solid", pt: 1, color: HEX.dk1 }, valign: "middle", margin: 0.08, objectName: "reward versions table" });
    s.addChart(pres.charts.BAR, [{ name: "Steps to first full lap", labels: ["Version 1", "Version 3"], values: [75000, 18000] }], {
      x: M, y: 4.15, w: 6.9, h: 2.7, barDir: "bar",
      showTitle: true, title: "Practice steps before the first full lap", titleColor: HEX.lt1, titleFontSize: 14,
      titleFontFace: "+mn-lt",
      chartColors: [HEX.accent3, HEX.accent4], invertedColors: [HEX.accent3, HEX.accent4],
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: HEX.lt1, dataLabelFontSize: 12,
      dataLabelFormatCode: "#,##0", dataLabelFontFace: "+mn-lt",
      catAxisLabelColor: HEX.lt2, valAxisLabelColor: HEX.lt2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "3A4350", size: 0.5 }, catGridLine: { style: "none" },
      valAxisHidden: true, catAxisLineShow: false, valAxisMaxVal: 95000,
      showLegend: false, barGapWidthPct: 60, objectName: "steps to first lap chart",
    });
    const vx = M + 7.25, vw = W - M - vx;
    const h = await video(s, "ch2_18k", vx, 1.5, vw, "version 3 video");
    caption(s, vx, 1.5 + h + 0.2, vw, [
      { text: "Version 3 at only 18,000 steps", options: { bold: true, color: C.accent1, breakLine: true } },
      { text: "Making the finish worth 4× more taught it about 4× faster." },
    ], "version 3 caption", 1.2);
    s.addText("Engineers call this reward tuning: change the points, retrain, watch what happens.", {
      x: vx, y: 1.5 + h + 1.5, w: vw, h: 1.0, isTextBox: true, fontSize: 16, italic: true, color: C.background2, margin: 0, valign: "top" });
  }
  s.addNotes("Version 2 added waypoints and doubled the waypoint bonus. It still parked, because finishing was only worth +20. Version 3 made finishing worth +80, and the robot learned the full lap in 18,000 steps instead of 75,000. Same robot, different points.");

  // 15 ─ Chapter 3 divider
  pres.addSection({ title: "Chapter 3: Random paths" });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Chapter 3: Random paths" });
  s.addText("CHAPTER 3", { placeholder: "kicker" });
  s.addText("Random paths, no memorizing", { placeholder: "title" });
  s.addText("So far it learned ONE path by heart. Real autos change every match, so every attempt now gets a brand-new random path. Our first big try failed. This chapter is how we fixed it.", { placeholder: "body" });
  s.addNotes("Memorizing one route is like memorizing one test's answers. We want it to actually understand driving to a point. Spoiler: it works now, but it took a failed training run and some detective work to get there.");

  // 16 ─ Random paths: the finished policy
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Every attempt is a new puzzle", { placeholder: "title" });
  {
    const lw = 4.6;
    const pts = [
      { Comp: fa.FaRandom, head: "Random every time", t: "3 to 12 waypoints, each 0.5 to 6 m from the last. Some legs pass right behind a hub or trench." },
      { Comp: fa.FaBrain, head: "No memorizing", t: "It can't learn one route by heart. It has to learn HOW to drive to any point." },
      { Comp: fa.FaFlagCheckered, head: "It works", t: "The finished robot completes 90% of brand-new routes and steers around obstacles by itself." },
    ];
    for (let i = 0; i < pts.length; i++) {
      const y = 1.5 + i * 1.7;
      await badge(s, pts[i].Comp, M, y, 0.8, HEX.accent1, HEX.dk1, pts[i].head);
      s.addText([
        { text: pts[i].head, options: { bold: true, fontSize: 18, color: C.background1, breakLine: true } },
        { text: pts[i].t, options: { fontSize: 15 } },
      ], { x: M + 1.05, y: y - 0.05, w: lw - 1.05, h: 1.5, isTextBox: true, color: C.background2, valign: "top", margin: 0 });
    }
    const vx = M + lw + 0.35, vw = W - M - vx;
    const h = await video(s, "ch3_final", vx, 1.5, vw, "final policy video");
    caption(s, vx, 1.5 + h + 0.25, vw, [
      { text: `${steps(manifest.ch3_final.train_step)} practice steps: `, options: { bold: true, color: C.accent1 } },
      { text: "all 10 waypoints in 8.8 seconds, going around two obstacles on the way. The yellow dot is the current target; the red dots are the start and finish." },
    ], "final policy caption", 1.2);
  }
  s.addNotes("Play the video. Each yellow dot is the current target waypoint, and the path is different every attempt. Watch it swing around the hub instead of driving into it. This is the policy we're bringing to our real robot code. But the first time we trained this, it didn't work at all.");

  // 16a ─ The failed run
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Our first try failed", { placeholder: "title" });
  {
    const vw = 6.0;
    const h = await video(s, "ch3_fail", M, 1.5, vw, "failed run video");
    caption(s, M, 1.5 + h + 0.2, vw, [
      { text: "500,000 steps in: ", options: { bold: true, color: C.accent2 } },
      { text: "it drives into a wall in 2.6 seconds. Every test drive from 100,000 steps on ended in a crash." },
    ], "failed run caption", 0.9);
    const failed = charts.failed || [];
    const cx = M + vw + 0.45, cw = W - M - cx;
    s.addChart(pres.charts.LINE, [{ name: "Average score", labels: failed.map((p) => p.k >= 1000 ? (p.k / 1000) + "M" : p.k + "k"),
      values: failed.map((p) => p.reward) }], {
      x: cx, y: 1.45, w: cw, h: 3.55, objectName: "failed run chart",
      showTitle: true, title: "Average score per attempt (Oct 3 run)", titleColor: HEX.lt1, titleFontSize: 14, titleFontFace: "+mn-lt",
      chartColors: [HEX.accent2], lineSize: 2.5, lineDataSymbol: "none",
      valAxisMinVal: -100, valAxisMaxVal: 0, valAxisMajorUnit: 25,
      valAxisLabelColor: HEX.lt2, catAxisLabelColor: HEX.lt2, valAxisLabelFontSize: 11, catAxisLabelFontSize: 11,
      valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", catAxisLabelFrequency: 4,
      valGridLine: { color: "3A4250", size: 0.5 }, catGridLine: { style: "none" },
      catAxisLineShow: false, valAxisLineShow: false, showLegend: false,
      showCatAxisTitle: true, catAxisTitle: "training steps", catAxisTitleColor: HEX.accent5, catAxisTitleFontSize: 11,
    });
    const stats = [
      { big: "500,000", small: "practice steps (about 2 hours on a laptop)", col: HEX.lt1 },
      { big: "0.2", small: "waypoints reached per attempt", col: HEX.accent2 },
      { big: "0", small: "routes finished in testing", col: HEX.accent2 },
    ];
    const sw = (W - 2 * M - 0.6) / 3, sy = 5.35, sh = 1.4;
    stats.forEach((st, i) => {
      const x = M + i * (sw + 0.3);
      card(s, x, sy, sw, sh, `failed stat ${i + 1}`);
      s.addText(st.big, { x: x + 0.3, y: sy + 0.1, w: sw - 0.6, h: 0.75, isTextBox: true, fontSize: 36, bold: true,
        color: st.col, valign: "middle", margin: 0 });
      s.addText(st.small, { x: x + 0.3, y: sy + 0.85, w: sw - 0.6, h: 0.45, isTextBox: true, fontSize: 14,
        color: C.background2, valign: "top", margin: 0 });
    });
  }
  s.addNotes("October 3: we trained the random-path robot for half a million steps. The score line is flat: after 500,000 steps it was no better than at the start. Every test video ended in a crash within a couple of seconds. It reached about one waypoint every five attempts. So: what went wrong?");

  // 16a2 ─ Detective work
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Detective work", { placeholder: "title" });
  {
    const cols = [
      { Comp: fa.FaSearch, fill: HEX.accent3, ic: HEX.lt1, head: "Theory: crashing was cheaper",
        t: "Running out the 60 s clock cost −90 points and a crash only −75. Maybe it crashed on purpose?",
        verdict: "Wrong. It only plans about 2 seconds ahead, and it never saw the crash coming.", vc: HEX.accent2 },
      { Comp: fa.FaRobot, fill: HEX.accent3, ic: HEX.lt1, head: "Test: a 3-line robot",
        t: "We wrote a simple program with no AI: drive straight at the yellow dot.",
        verdict: "It scored +214 per attempt; the AI scored −67. The points were fine, the learning was broken.", vc: HEX.accent4 },
      { Comp: fa.FaCompressArrowsAlt, fill: HEX.accent3, ic: HEX.lt1, head: "Shrink the problem",
        t: "One waypoint, no walls to crash into: it learned in minutes. Then add pieces back one at a time.",
        verdict: "Each piece that stopped the learning was a bug to fix. Next slide.", vc: HEX.accent1 },
    ];
    const cw = 3.85, gap = (W - 2 * M - 3 * cw) / 2, y = 1.45, ch = 4.1;
    for (let i = 0; i < cols.length; i++) {
      const c = cols[i], x = M + i * (cw + gap);
      card(s, x, y, cw, ch, c.head + " card");
      await badge(s, c.Comp, x + 0.3, y + 0.3, 0.8, c.fill, c.ic, c.head);
      s.addText(c.head, { x: x + 1.25, y: y + 0.3, w: cw - 1.5, h: 0.8, isTextBox: true, fontSize: 18,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
      s.addText(c.t, { x: x + 0.3, y: y + 1.3, w: cw - 0.6, h: 1.25, isTextBox: true, fontSize: 15,
        color: C.background2, valign: "top", margin: 0 });
      s.addText(c.verdict, { x: x + 0.3, y: y + 2.65, w: cw - 0.6, h: 1.3, isTextBox: true, fontSize: 15,
        bold: true, color: c.vc, valign: "top", margin: 0 });
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.85, w: W - 2 * M, h: 0.85, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.5 }, objectName: "takeaway outline" });
    s.addText([
      { text: "Measure, don't guess. ", options: { bold: true, color: C.background1 } },
      { text: "Our first explanation sounded right, and a quick test proved it wrong." },
    ], { x: M + 0.35, y: 5.85, w: W - 2 * M - 0.7, h: 0.85, isTextBox: true, fontSize: 17,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("Our first guess: crashing was cheaper than timing out, so it crashed on purpose. Sounds right! But the robot only cares about roughly the next 2 seconds of points (that's what the 0.99 'discount' setting does), so a 60-second clock barely mattered to it, and when we looked inside its brain, it predicted good things right up until it hit the wall. Then the key test: a three-line program that just drives at the dot. It beat the AI by almost 300 points per attempt. That told us the scoring was fine; the AI just wasn't learning. Finally we shrank the problem down to one waypoint and added pieces back until learning broke.");

  // 16a3 ─ What was broken
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("What was broken, and the fix", { placeholder: "title" });
  {
    const items = [
      { Comp: fa.FaBug, head: "A crash looked like a time-out",
        p: "The code told the AI “time ran out” when it crashed, so it assumed it would have kept driving.",
        f: "A crash now ends the attempt for real." },
      { Comp: fa.FaSearchMinus, head: "Numbers too tiny to read",
        p: "Waypoint directions were divided by the 18 m field, so a 2 m trip read as 0.11.",
        f: "Divide by 6 m instead." },
      { Comp: fa.FaHourglassHalf, head: "60 seconds to get lost",
        p: "A lost robot wandered for 3,000 steps, so it rarely reached anything to learn from.",
        f: "Give up after 5 s per waypoint, like our robot code." },
      { Comp: fa.FaGraduationCap, head: "The final exam on day one",
        p: "Up to 12 random waypoints, some behind obstacles, from the very first step.",
        f: "A curriculum: short, clear paths first." },
      { Comp: fa.FaShieldAlt, head: "Too scared to move",
        p: "Once crashes really ended the attempt, −75 made sitting still look safest.",
        f: "−10. A crash already loses every point it could have earned." },
      { Comp: fa.FaEyeSlash, head: "Blind to obstacles",
        p: "It had to memorize where the hub is from its position alone.",
        f: "8 distance rays so it can SEE obstacles." },
    ];
    const cw = (W - 2 * M - 0.6) / 3, ch = 2.5, gy = 0.3, y0 = 1.45;
    for (let i = 0; i < items.length; i++) {
      const it = items[i], x = M + (i % 3) * (cw + 0.3), y = y0 + Math.floor(i / 3) * (ch + gy);
      card(s, x, y, cw, ch, it.head + " card");
      await badge(s, it.Comp, x + 0.3, y + 0.25, 0.7, HEX.accent2, HEX.lt1, it.head);
      s.addText(it.head, { x: x + 1.15, y: y + 0.25, w: cw - 1.4, h: 0.7, isTextBox: true, fontSize: 17,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
      s.addText([
        { text: it.p, options: { breakLine: true } },
        { text: "Fix: ", options: { bold: true, color: C.accent4 } },
        { text: it.f, options: { color: C.background1 } },
      ], { x: x + 0.3, y: y + 1.05, w: cw - 0.6, h: 1.35, isTextBox: true, fontSize: 14,
        color: C.background2, valign: "top", margin: 0, paraSpaceAfter: 6 });
    }
  }
  s.addNotes("Six problems, all found by the shrink-the-problem test or by watching the robot. The first one is a real bug students can learn from: in our code, 'crashed' and 'ran out of time' used the same signal, and the AI library treats 'ran out of time' as 'you would have kept going'. So it never learned that crashes end everything. Later we found the same bug on the 5-second give-up rule and fixed it too. 'Too scared to move' is not the same as the theory on the last slide: that theory said crashing was too cheap. After fix 1, crashes really did end everything, and then −75 was so painful that the one-waypoint test learned to sit still instead of steering. The curriculum is the same idea as Chapter 1: start easy. Each fix is its own commit in our git history, with the reason written down.");

  // 16b ─ Observations then vs now
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("What it sees: then vs. now", { placeholder: "title" });
  {
    const cw = 1.22, gap = 0.14, ch = 0.7;
    const x0 = M;
    s.addText("Version 1 (figure-8): 9 numbers", { x: x0, y: 1.4, w: 9, h: 0.45, isTextBox: true, fontSize: 20, bold: true,
      color: C.accent1, margin: 0 });
    chipRow(s, OBS_V1, x0, 2.0, cw, ch, gap, "Version 1", { dropped: new Set([6, 7, 8]) });
    s.addText([
      { text: "Crossed out: ", options: { bold: true, color: C.background1 } },
      { text: "“Progress” and “Off line” only make sense with ONE fixed path. Heading isn't needed while the robot never turns." },
    ], { x: x0, y: 2.85, w: W - 2 * M, h: 0.6, isTextBox: true, fontSize: 16, color: C.background2, margin: 0, valign: "top" });

    s.addText("Now (random paths): 16 numbers, 10 of them new", { x: x0, y: 3.6, w: 9, h: 0.45, isTextBox: true, fontSize: 20, bold: true,
      color: C.accent1, margin: 0 });
    chipRow(s, OBS_NOW, x0, 4.3, cw, ch, gap, "Now", { isNew: new Set([2, 3]) });
    chipRow(s, OBS_RAYS, x0, 5.35, cw, ch, gap, "Rays", { isNew: new Set([0, 1, 2, 3, 4, 5, 6, 7]) });
    s.addText([
      { text: "New: ", options: { bold: true, color: C.accent4 } },
      { text: "Field X and Y, where it is (added in June). " },
      { text: "8 rays", options: { bold: true, color: "A78BFA" } },
      { text: ", how far it can drive in each compass direction before hitting something (added in October)." },
    ], { x: x0, y: 6.2, w: W - 2 * M, h: 0.6, isTextBox: true, fontSize: 16, color: C.background2, margin: 0, valign: "top" });
  }
  s.addNotes("Top row: the 9 numbers from Chapter 1. Three only made sense for one fixed path, so they're gone. Bottom: the 16 numbers the robot sees today: 6 kept from version 1 and 10 new ones. Field X and Y came first, back in June (next slide). The 8 rays came in October as one of the fixes (the slide after that). Ray E, NE, N... are compass directions on the field: E points toward the red alliance wall. The waypoint numbers are also scaled differently now: divided by 6 m instead of the 18 m field, one of the fixes from the last slide.");

  // 16c ─ Why position
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Why we added position", { placeholder: "title" });
  {
    const cw = 5.85, ch = 3.3, y = 1.5;
    const cols = [
      { x: M, Comp: fa.FaLowVision, fill: HEX.accent2, ic: HEX.lt1, head: "Without position",
        t: "Imagine walking through a dark room with a compass that only points to the door. You bump into the same table every time, but you can't learn where it is, because you don't know where YOU are." },
      { x: W - M - cw, Comp: fa.FaMapMarkerAlt, fill: HEX.accent4, ic: HEX.dk1, head: "With position",
        t: "Now you have a map with a dot showing where you are. “I always bump into something right HERE” is something you can learn. The hub is always in the same spot." },
    ];
    for (const c of cols) {
      card(s, c.x, y, cw, ch, c.head + " card");
      await badge(s, c.Comp, c.x + 0.35, y + 0.35, 0.95, c.fill, c.ic, c.head);
      s.addText(c.head, { x: c.x + 1.55, y: y + 0.35, w: cw - 1.9, h: 0.95, isTextBox: true, fontSize: 24,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
      s.addText(c.t, { x: c.x + 0.35, y: y + 1.5, w: cw - 0.7, h: 1.7, isTextBox: true, fontSize: 16,
        color: C.background2, valign: "top", margin: 0 });
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.1, w: W - 2 * M, h: 1.6, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.5 }, objectName: "quote outline" });
    await badge(s, fa.FaLightbulb, M + 0.35, 5.45, 0.8, HEX.accent1, HEX.dk1, "next idea");
    s.addText([
      { text: "But a map only works for ONE field. ", options: { bold: true, color: C.background1 } },
      { text: "Position (added in June) made it memorize where the 2026 hub is, and next year the field changes. So in October we gave it something better: eyes." },
    ], { x: M + 1.45, y: 5.1, w: W - 2 * M - 1.8, h: 1.6, isTextBox: true, fontSize: 17,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("In version 1 the path was drawn around the obstacles for it, so it didn't need to know where they were. With random paths, a waypoint can be on the far side of a hub, so the robot has to know where the hub is. Position made that learnable, but only by memorizing this one field. That's slow to learn and useless in 2027. Next slide: how we let it see obstacles instead.");

  // 16d ─ Distance rays
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Seeing obstacles: 8 distance rays", { placeholder: "title" });
  {
    const ox = 4.55, oy = 2.35, ow = 1.9, oh = 2.2;   // hub
    s.addShape(pres.shapes.RECTANGLE, { x: ox, y: oy, w: ow, h: oh, fill: { color: HEX.accent2 },
      line: { color: HEX.lt1, width: 1 }, objectName: "hub" });
    s.addText("HUB", { x: ox, y: oy, w: ow, h: oh, isTextBox: true, align: "center", valign: "middle",
      fontSize: 22, bold: true, color: HEX.lt1, margin: 0 });
    const rs = 0.7, cx = 2.55, cy = oy + oh / 2, L = 1.75;
    const dirs = [["E", 1, 0], ["NE", 1, -1], ["N", 0, -1], ["NW", -1, -1], ["W", -1, 0], ["SW", -1, 1], ["S", 0, 1], ["SE", 1, 1]];
    for (const [nm, dx, dy] of dirs) {
      const n = Math.hypot(dx, dy), ux = dx / n, uy = dy / n;
      const hit = nm === "E";
      const len = hit ? (ox - cx) : L;                      // the east ray stops at the hub
      const x1 = cx + ux * rs * 0.55, y1 = cy + uy * rs * 0.55, x2 = cx + ux * len, y2 = cy + uy * len;
      s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
        flipH: x2 < x1, flipV: y2 < y1, line: { color: hit ? HEX.accent1 : HEX.accent4, width: hit ? 3.5 : 2.5,
          dashType: hit ? "solid" : "dash" }, objectName: `ray ${nm}` });
      if (!hit) {
        s.addShape(pres.shapes.OVAL, { x: x2 - 0.07, y: y2 - 0.07, w: 0.14, h: 0.14, fill: { color: HEX.accent4 },
          line: { type: "none" }, objectName: `ray ${nm} end` });
      }
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx - rs / 2, y: cy - rs / 2, w: rs, h: rs, rectRadius: 0.08,
      fill: { color: HEX.lt1 }, line: { color: HEX.accent4, width: 2 }, objectName: "robot" });
    s.addImage({ data: await icon(fa.FaRobot, HEX.dk1), x: cx - rs / 2 + 0.14, y: cy - rs / 2 + 0.14, w: rs - 0.28, h: rs - 0.28,
      objectName: "robot icon", altText: "robot" });
    s.addText("short!", { x: ox - 1.1, y: cy - 0.38, w: 1.0, h: 0.3, isTextBox: true, fontSize: 13, align: "right",
      bold: true, color: C.accent1, margin: 0 });
    s.addText("Yellow ray: short, the hub is close. Green rays: nothing within 2 m. (Drawn bigger so you can see it.)", { x: M, y: oy + oh + 1.05, w: 6.9, h: 0.6,
      isTextBox: true, fontSize: 12, italic: true, color: C.accent5, margin: 0 });
    const sx = 7.75, sw = W - M - sx;
    card(s, sx, 1.45, sw, 5.3, "rays explainer card");
    s.addText([
      { text: "Each ray asks: how far can I drive this way before my bumper hits something? (up to 2 m)", options: { breakLine: true } },
      { text: "Like a car's parking sensors, pointing 8 ways at once.", options: { italic: true, color: C.accent1, breakLine: true } },
      { text: "It works on ANY shape: 2025's hexagon, 2026's rectangles, whatever 2027 brings.", options: { bold: true, color: C.background1, breakLine: true } },
      { text: "We removed the old warning zone (a growing penalty within 15 cm of an obstacle). With rays it didn't help, so we kept the simpler reward." },
    ], { x: sx + 0.35, y: 1.7, w: sw - 0.7, h: 4.85, isTextBox: true, fontSize: 17, color: C.background2,
      valign: "top", margin: 0, paraSpaceAfter: 14 });
  }
  s.addNotes("Each ray is one number between 0 (touching) and 1 (clear for at least 2 m). In the picture the east ray is short because the hub is right there. On the robot, these get computed from the corners of each field element, which our 1507Base code already stores, so next year we only swap in the new field's shapes. Earlier we tried a warning zone: a growing penalty within 15 cm of an obstacle. With rays the robot can see obstacles directly. We ran the same 100k-step training with and without the warning zone: without it, the robot reached the second curriculum stage sooner (44k vs 53k steps) and timed out less (24% vs 32%). That's one run each, so it's a hint, not proof, but when two versions tie, keep the simpler one.");

  // 16e ─ Today's scoreboard
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Today's scoreboard", { placeholder: "title" });
  {
    const items = [
      { p: "+2", u: "per meter", t: "closer to the waypoint, measured along the path around obstacles", c: HEX.accent4 },
      { p: "+0.8", u: "every step", t: "for driving toward the waypoint at full speed", c: HEX.accent4 },
      { p: "+100", u: "", t: "for reaching a waypoint (within 0.40 m)", c: HEX.accent4 },
      { p: "+75", u: "", t: "for finishing every waypoint", c: HEX.accent4 },
      { p: "−0.03", u: "every step", t: "about −1.5 per second, so keep moving", c: HEX.accent2 },
      { p: "−10", u: "", t: "for a crash, and the attempt ends", c: HEX.accent2 },
      { p: "5 s", u: "", t: "to reach each waypoint, or the attempt ends", c: HEX.accent1 },
      { p: "0", u: "", t: "for wiggling: backing up gives back what moving closer earned", c: HEX.accent5 },
    ];
    const cols = 4, gx = 0.25, cw = (W - 2 * M - 3 * gx) / 4, ch = 1.6, gy = 0.2, y0 = 1.4;
    items.forEach((it, i) => {
      const x = M + (i % cols) * (cw + gx), y = y0 + Math.floor(i / cols) * (ch + gy);
      card(s, x, y, cw, ch, `score card ${i + 1}`);
      s.addText([
        { text: it.p, options: { bold: true, fontSize: 28, color: it.c } },
        { text: it.u ? "  " + it.u : "", options: { fontSize: 13, color: HEX.accent5 } },
      ], { x: x + 0.25, y: y + 0.1, w: cw - 0.5, h: 0.6, isTextBox: true, valign: "middle", margin: 0 });
      s.addText(it.t, { x: x + 0.25, y: y + 0.72, w: cw - 0.5, h: 0.8, isTextBox: true, fontSize: 14,
        color: C.background2, valign: "top", margin: 0 });
    });
    // Detour diagram: straight line vs. path around the hub
    const by = 5.1, bh = 1.65;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: by, w: W - 2 * M, h: bh, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.5 }, objectName: "detour box" });
    const ry = by + bh / 2, rx = M + 0.45, tx = M + 3.85, hx = M + 1.75, hw = 0.75, hh = 0.85;
    s.addShape(pres.shapes.RECTANGLE, { x: hx, y: ry - hh / 2 + 0.12, w: hw, h: hh, fill: { color: HEX.accent2 },
      line: { type: "none" }, objectName: "detour hub" });
    s.addShape(pres.shapes.LINE, { x: rx, y: ry, w: tx - rx, h: 0, line: { color: HEX.accent2, width: 2, dashType: "dash" },
      objectName: "straight path" });
    const top = ry - hh / 2 - 0.1;
    const pts = [[rx, ry], [hx - 0.08, top], [hx + hw + 0.08, top], [tx, ry]];
    for (let i = 0; i < pts.length - 1; i++) {
      const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
      s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
        h: Math.abs(y2 - y1), flipV: y2 < y1 && x2 > x1 ? true : false, flipH: false,
        line: { color: HEX.accent4, width: 3, endArrowType: i === pts.length - 2 ? "triangle" : undefined },
        objectName: `around path ${i + 1}` });
    }
    s.addShape(pres.shapes.OVAL, { x: rx - 0.14, y: ry - 0.14, w: 0.28, h: 0.28, fill: { color: HEX.lt1 },
      line: { type: "none" }, objectName: "detour robot" });
    s.addShape(pres.shapes.OVAL, { x: tx - 0.12, y: ry - 0.12, w: 0.24, h: 0.24, fill: { color: HEX.accent1 },
      line: { type: "none" }, objectName: "detour target" });
    s.addText([
      { text: "Going around counts as progress. ", options: { bold: true, color: C.background1 } },
      { text: "We used to measure the straight line (red). Swinging around the hub made that line longer, so the robot lost points for doing the right thing and learned to stall. Now we measure the green path." },
    ], { x: M + 4.45, y: by, w: W - 2 * M - 4.8, h: bh, isTextBox: true, fontSize: 15,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("The whole reward for the working robot. The two per-step rewards are the same pair the figure-8 robot from Chapter 1 learned from: get closer, and point your speed at the target. Because 'closer' is a change in distance, driving away gives back exactly what driving closer earned, so wiggling can't farm points (an earlier version of the robot found that loophole). The big idea at the bottom: distance is measured along the shortest path around the obstacles, not in a straight line. With the straight line, the correct detour looked like going the wrong way.");

  // 16g ─ The breakthrough (learning curve)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("The breakthrough", { placeholder: "title" });
  {
    const rt = charts.route_test || [];
    // label only round values (500k, 1M, 1.5M, ...) so the axis reads cleanly
    // ...plus the first point, so students can see the chart starts at 150k, not 0
    const labels = rt.map((p, i) => (p.k % 500 && i > 0) ? "" : p.k < 1000 ? p.k + "k" : (p.k / 1000) + "M");
    const cw = 8.3;
    s.addChart(pres.charts.LINE, [
      { name: "Finished", labels, values: rt.map((p) => p.complete) },
      { name: "Crashed", labels, values: rt.map((p) => p.crash) },
    ], {
      x: M, y: 1.4, w: cw, h: 4.95, objectName: "learning curve chart",
      showTitle: true, title: "% of 50 hard test routes, every 50,000 steps", titleColor: HEX.lt1, titleFontSize: 14, titleFontFace: "+mn-lt",
      chartColors: [HEX.accent4, HEX.accent2], lineSize: 2.5, lineDataSymbol: "none",
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20,
      valAxisLabelColor: HEX.lt2, catAxisLabelColor: HEX.lt2, valAxisLabelFontSize: 11, catAxisLabelFontSize: 11,
      valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", catAxisLabelFrequency: 1,
      valGridLine: { color: "3A4250", size: 0.5 }, catGridLine: { style: "none" },
      catAxisLineColor: HEX.accent5, valAxisLineShow: false,
      showLegend: true, legendPos: "b", legendColor: HEX.lt2, legendFontSize: 12, legendFontFace: "+mn-lt",
      showCatAxisTitle: true, catAxisTitle: "training steps", catAxisTitleColor: HEX.accent5, catAxisTitleFontSize: 11,
    });
    s.addText("Starts at 150,000 steps: before that, the robot was still practicing on the easy curriculum stages.", {
      x: M, y: 6.4, w: cw, h: 0.4, isTextBox: true, fontSize: 12, italic: true, color: C.accent5, margin: 0 });
    const notes = [
      { head: "Stuck around 40%", sub: "250k to 550k steps. It looked like it had stopped learning.", c: HEX.accent1 },
      { head: "Breakthrough", sub: "550k to 1M: finished routes climbed and crashes fell toward zero.", c: HEX.accent4 },
      { head: "Overnight", sub: "We paused at 1M and kept training overnight. A short dip, then it leveled off around 90%.", c: HEX.accent3 },
    ];
    const nx = M + cw + 0.35, nw = W - M - nx, nh = 1.6;
    for (let i = 0; i < notes.length; i++) {
      const y = 1.45 + i * (nh + 0.2);
      card(s, nx, y, nw, nh, notes[i].head + " card");
      s.addText([
        { text: notes[i].head, options: { bold: true, fontSize: 18, color: notes[i].c, breakLine: true } },
        { text: notes[i].sub, options: { fontSize: 14 } },
      ], { x: nx + 0.25, y: y + 0.15, w: nw - 0.5, h: nh - 0.3, isTextBox: true, color: C.background2,
        valign: "top", margin: 0, paraSpaceAfter: 4 });
    }
  }
  s.addNotes("Every 10,000 steps the trainer drives the same 50 hard routes and records how many it finishes; this chart averages those tests over every 50,000 steps. It starts at 150,000 because the hard routes only began at 110,000 steps, once the robot had passed the two easy curriculum stages. For 400,000 steps it hovered around 40% and we wondered if it was stuck. Then it took off. The dip right after 1M is from restarting training, which starts with an empty memory of past attempts; it recovered within about 150,000 steps. Total: 2.5 million steps, about 14 hours of simulated driving, which took about 10 hours of computer time (8 PM to 6:30 AM): the simulator runs faster than real time. Lesson: some learning looks flat for a long time before it clicks.");

  // 16h ─ Same route, getting better
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Same route, getting better", { placeholder: "title" });
  {
    const vw = 4.7, gxv = 0.5, x0 = (W - 2 * vw - gxv) / 2;
    const vids = [
      { key: "ch3_150k", t: "crashes in under a second", c: C.accent2 },
      { key: "ch3_200k", t: "gets most of the way, then gets stuck", c: C.accent1 },
      { key: "ch3_500k", t: "almost there, crashes near the end", c: C.accent2 },
      { key: "ch3_1m", t: "all 10 waypoints in 8.7 seconds", c: C.accent4 },
    ];
    for (let i = 0; i < vids.length; i++) {
      const v = vids[i], x = x0 + (i % 2) * (vw + gxv), y = 1.45 + Math.floor(i / 2) * 2.7;
      const h = await video(s, v.key, x, y, vw, v.key + " video");
      s.addText([
        { text: steps(manifest[v.key].train_step) + " steps: ", options: { bold: true, color: v.c } },
        { text: v.t },
      ], { x, y: y + h + 0.1, w: vw, h: 0.4, isTextBox: true, fontSize: 15, color: C.background2, margin: 0, valign: "top" });
    }
  }
  s.addNotes("These four videos are the SAME route: 10 waypoints, two of them behind obstacles. We fixed the test route on purpose. Before that, each test video used a new random route, and a lucky easy route at one checkpoint made a later checkpoint on a hard route look like the robot was getting worse. Ask: what changes between 200,000 and 1,000,000 steps?");

  // 16i ─ Final result
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("The final result", { placeholder: "title" });
  {
    const stats = [
      { big: "90%", small: "of brand-new routes finished", c: HEX.accent4 },
      { big: "5%", small: "crashed", c: HEX.accent2 },
      { big: "96%", small: "of legs around a hub or trench made it", c: HEX.accent1 },
      { big: "2.9", small: "m/s average speed", c: HEX.accent3 },
    ];
    const gx = 0.3, sw = (W - 2 * M - 3 * gx) / 4, sy = 1.5, sh = 2.25;
    stats.forEach((st, i) => {
      const x = M + i * (sw + gx);
      card(s, x, sy, sw, sh, `result stat ${i + 1}`);
      s.addText(st.big, { x: x + 0.25, y: sy + 0.2, w: sw - 0.5, h: 1.15, isTextBox: true, fontSize: 60, bold: true,
        color: st.c, valign: "middle", margin: 0 });
      s.addText(st.small, { x: x + 0.25, y: sy + 1.4, w: sw - 0.5, h: 0.75, isTextBox: true, fontSize: 15,
        color: C.background2, valign: "top", margin: 0 });
    });
    const rows = [
      { Comp: fa.FaTimesCircle, fill: HEX.accent2, ic: HEX.lt1, head: "October 3",
        t: "500,000 steps, 0.2 waypoints per attempt, every test drive crashed." },
      { Comp: fa.FaCheckCircle, fill: HEX.accent4, ic: HEX.dk1, head: "October 5",
        t: "2.5 million steps, 90% of hard routes finished, ready to try on our robot code." },
    ];
    const rw = (W - 2 * M - 0.3) / 2, ry = 4.1, rh = 1.55;
    for (let i = 0; i < rows.length; i++) {
      const x = M + i * (rw + 0.3);
      card(s, x, ry, rw, rh, rows[i].head + " card");
      await badge(s, rows[i].Comp, x + 0.3, ry + 0.35, 0.85, rows[i].fill, rows[i].ic, rows[i].head);
      s.addText([
        { text: rows[i].head, options: { bold: true, fontSize: 20, color: C.background1, breakLine: true } },
        { text: rows[i].t, options: { fontSize: 15 } },
      ], { x: x + 1.4, y: ry + 0.15, w: rw - 1.7, h: rh - 0.3, isTextBox: true, color: C.background2, valign: "middle", margin: 0 });
    }
    s.addText("Tested on 150 routes it never saw while we picked the best model. 2.5 million steps is about 14 hours of simulated driving.", {
      x: M, y: 5.9, w: W - 2 * M, h: 0.7, isTextBox: true, fontSize: 13, italic: true, color: C.accent5, margin: 0, valign: "top" });
  }
  s.addNotes("These numbers come from 150 brand-new routes the robot never saw while we were choosing which saved brain to keep. That matters: if you test on the same routes you used to pick the winner, the winner looks better than it is. Two days: from crashing every time to finishing 9 out of 10 hard routes.");

  // 16j ─ Lessons
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("What the failure taught us", { placeholder: "title" });
  {
    const items = [
      { Comp: fa.FaSearch, head: "Measure, don't guess", t: "Our first theory sounded right and was wrong. A quick test told us." },
      { Comp: fa.FaRobot, head: "Compare to something simple", t: "A 3-line program beat the AI. That proved the points were fine and the learning was broken." },
      { Comp: fa.FaLayerGroup, head: "Easy first, then harder", t: "Short, clear paths before random ones, the same way you'd learn a new skill." },
      { Comp: fa.FaVideo, head: "One video isn't data", t: "A single test drive can be lucky. We test 50 routes and keep the best brain, not the last one." },
    ];
    const cw = (W - 2 * M - 0.3) / 2, ch = 2.45;
    for (let i = 0; i < items.length; i++) {
      const x = M + (i % 2) * (cw + 0.3), y = 1.5 + Math.floor(i / 2) * (ch + 0.3);
      card(s, x, y, cw, ch, items[i].head + " card");
      await badge(s, items[i].Comp, x + 0.35, y + 0.35, 0.9, HEX.accent1, HEX.dk1, items[i].head);
      s.addText(items[i].head, { x: x + 1.5, y: y + 0.35, w: cw - 1.8, h: 0.9, isTextBox: true, fontSize: 22,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
      s.addText(items[i].t, { x: x + 0.35, y: y + 1.4, w: cw - 0.7, h: 0.9, isTextBox: true, fontSize: 16,
        color: C.background2, valign: "top", margin: 0 });
    }
  }
  s.addNotes("These apply to any engineering problem, not just AI. 'Keep the best, not the last': the robot's skill swung up and down by 10 to 20 points between saves, so the trainer now tests every 10,000 steps and keeps the best one. All of this is in our git history with the reasons written down, so you can read exactly what we tried.");

  // 16f ─ Where we're going next
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Where we're going next", { placeholder: "title" });
  {
    s.addText("Every observation is a question the robot can ask itself. We add one only when it needs the answer.", {
      x: M, y: 1.35, w: W - 2 * M, h: 0.5, isTextBox: true, fontSize: 18, italic: true, color: C.accent1, margin: 0 });
    const items = [
      { Comp: fa.FaSyncAlt, head: "“Which way am I facing?”", t: "Bring heading back so the robot can turn and face a target, like the hub when it shoots." },
      { Comp: fa.FaGamepad, head: "“What does the driver want?”", t: "Teleop assist: the policy sees the driver's joystick and helps them steer around obstacles." },
      { Comp: fa.FaGlobeAmericas, head: "“What does 2027's field look like?”", t: "Train with random obstacles, so the rays work on any field, not just 2026's." },
      { Comp: fa.FaBalanceScale, head: "Just enough", t: "Too few numbers and it can't decide. Too many and it learns slower. Pick what matters." },
    ];
    const cw = (W - 2 * M - 0.3) / 2, ch = 2.2;
    for (let i = 0; i < items.length; i++) {
      const x = M + (i % 2) * (cw + 0.3), y = 2.05 + Math.floor(i / 2) * (ch + 0.3);
      card(s, x, y, cw, ch, items[i].head + " card");
      await badge(s, items[i].Comp, x + 0.35, y + 0.35, 0.85, HEX.accent4, HEX.dk1, items[i].head);
      s.addText(items[i].head, { x: x + 1.4, y: y + 0.35, w: cw - 1.7, h: 0.85, isTextBox: true, fontSize: 20,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
      s.addText(items[i].t, { x: x + 0.35, y: y + 1.3, w: cw - 0.7, h: 0.8, isTextBox: true, fontSize: 15,
        color: C.background2, valign: "top", margin: 0 });
    }
  }
  s.addNotes("Where the robot goes next. Right now it can't rotate, so it doesn't need heading; once it has to face the hub, heading comes back. Teleop assist is our other experiment. The simulator already moves like our 2027 robot: same top speed (5.04 m/s) and the same acceleration limits as our 1507Base code. Next step is a new field every season, which the rays make possible.");

  // 17 ─ Sim to robot
  pres.addSection({ title: "This season" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "This season" });
  s.addText("From the simulator to our robot", { placeholder: "title" });
  {
    const stages = [
      { Comp: fa.FaLaptopCode, fill: HEX.accent1, ic: HEX.dk1, head: "1. Train",
        sub: "Swerve Policy Playground", t: "Python simulator. The robot practices millions of steps." },
      { Comp: fa.FaVial, fill: HEX.accent3, ic: HEX.lt1, head: "2. Test",
        sub: "1507Labs", t: "Our real robot code in the WPILib (Worcester Polytechnic Institute Library) simulator. Does the policy drive our actual swerve code?" },
      { Comp: fa.FaMicrochip, fill: HEX.accent4, ic: HEX.dk1, head: "3. Drive",
        sub: "1507Base on SystemCore", t: "The 2027 robot controller runs autos using the policy as the driver." },
    ];
    const cw = 3.75, gap = (W - 2 * M - 3 * cw) / 2, y = 1.6, ch = 3.6;
    for (let i = 0; i < stages.length; i++) {
      const st = stages[i], x = M + i * (cw + gap);
      card(s, x, y, cw, ch, st.head + " card");
      await badge(s, st.Comp, x + 0.35, y + 0.35, 0.95, st.fill, st.ic, st.head);
      s.addText([
        { text: st.head, options: { bold: true, fontSize: 24, color: C.background1, breakLine: true } },
        { text: st.sub, options: { fontSize: 15, color: C.accent1 } },
      ], { x: x + 1.45, y: y + 0.3, w: cw - 1.6, h: 1.05, isTextBox: true, valign: "middle", margin: 0 });
      s.addText(st.t, { x: x + 0.35, y: y + 1.6, w: cw - 0.7, h: 1.8, isTextBox: true, fontSize: 16,
        color: C.background2, valign: "top", margin: 0 });
      if (i < stages.length - 1) {
        s.addShape(pres.shapes.LINE, { x: x + cw + 0.05, y: y + ch / 2, w: gap - 0.1, h: 0,
          line: { color: HEX.accent5, width: 2.5, endArrowType: "triangle" }, objectName: "pipeline arrow " + (i + 1) });
      }
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.6, w: W - 2 * M, h: 1.05, rectRadius: 0.12,
      fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.5 }, objectName: "takeaway outline" });
    s.addText([
      { text: "The trained brain is just a big list of numbers (weights). ", options: { bold: true, color: C.background1 } },
      { text: "We copy those numbers into Java, and the robot does the same math the simulator did." },
    ], { x: M + 0.35, y: 5.6, w: W - 2 * M - 0.7, h: 1.05, isTextBox: true, fontSize: 17,
      color: C.background2, valign: "middle", margin: 0 });
  }
  s.addNotes("WPILib is the Worcester Polytechnic Institute Library, the FRC robot software we already use. SystemCore is the new 2027 robot controller. 1507Base already has a slot for a policy driver in autos; this season we fill it.");

  // 18 ─ Your turn
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "This season" });
  s.addText("Your turn: design a reward", { placeholder: "title" });
  {
    card(s, M, 1.45, W - 2 * M, 1.2, "challenge card");
    await badge(s, fa.FaLightbulb, M + 0.3, 1.65, 0.8, HEX.accent1, HEX.dk1, "challenge");
    s.addText([
      { text: "Challenge: ", options: { bold: true, color: C.accent1 } },
      { text: "teach the robot to drive to the hub and STOP right in front of it, facing it." },
    ], { x: M + 1.35, y: 1.45, w: W - 2 * M - 1.6, h: 1.2, isTextBox: true, fontSize: 20,
      color: C.background1, valign: "middle", margin: 0 });
    const cols = [
      { head: "What earns points?", Comp: fa.FaPlusCircle, fill: HEX.accent4, ic: HEX.dk1 },
      { head: "What loses points?", Comp: fa.FaMinusCircle, fill: HEX.accent2, ic: HEX.lt1 },
      { head: "How could it cheat?", Comp: fa.FaUserSecret, fill: HEX.accent6, ic: HEX.dk1 },
    ];
    const cw = 3.85, gap = 0.3, y = 2.95, ch = 3.75;
    for (let i = 0; i < cols.length; i++) {
      const x = M + i * (cw + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, rectRadius: 0.12,
        fill: { color: HEX.dk1 }, line: { color: cols[i].fill, width: 1.5, dashType: "dash" }, objectName: cols[i].head + " box" });
      await badge(s, cols[i].Comp, x + 0.3, y + 0.3, 0.7, cols[i].fill, cols[i].ic, cols[i].head);
      s.addText(cols[i].head, { x: x + 1.15, y: y + 0.3, w: cw - 1.35, h: 0.7, isTextBox: true, fontSize: 19,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
    }
  }
  s.addNotes("Group activity, 5 to 10 minutes. Fill the three boxes on the whiteboard. The 'cheat' column is the most important: every reward has a loophole. Later this season we'll actually train the best ideas.");

  // 19 ─ What's next
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "This season" });
  s.addText("What we'll do this season", { placeholder: "title" });
  {
    const items = [
      { Comp: fa.FaSlidersH, head: "Tune rewards", t: "Change the points, retrain, and see if the robot does what you meant." },
      { Comp: fa.FaVideo, head: "Watch and predict", t: "Before each training video, guess what the robot will do. Then check." },
      { Comp: fa.FaRoute, head: "Drive real autos", t: "Get a policy driving our autonomous routines on the 2027 robot." },
      { Comp: fa.FaFlagCheckered, head: "Kickoff", t: "New game in January means new goals, new obstacles, and new rewards to design." },
    ];
    const cw = (W - 2 * M - 0.3) / 2, ch = 2.45;
    for (let i = 0; i < items.length; i++) {
      const x = M + (i % 2) * (cw + 0.3), y = 1.5 + Math.floor(i / 2) * (ch + 0.3);
      card(s, x, y, cw, ch, items[i].head + " card");
      await badge(s, items[i].Comp, x + 0.35, y + 0.35, 0.9, HEX.accent1, HEX.dk1, items[i].head);
      s.addText(items[i].head, { x: x + 1.5, y: y + 0.35, w: cw - 1.8, h: 0.9, isTextBox: true, fontSize: 22,
        bold: true, color: C.background1, valign: "middle", margin: 0 });
      s.addText(items[i].t, { x: x + 0.35, y: y + 1.4, w: cw - 0.7, h: 0.9, isTextBox: true, fontSize: 16,
        color: C.background2, valign: "top", margin: 0 });
    }
  }
  s.addNotes("Wrap up. Everyone gets to tune and train their own version. No math required to start, just ideas and patience.");

  // 20 ─ Glossary
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "This season" });
  s.addText("Glossary", { placeholder: "title" });
  {
    const g = [
      ["AI", "Artificial Intelligence", "Computers doing things that normally need human thinking"],
      ["ML", "Machine Learning", "Computers improving from data or experience instead of hand-written rules"],
      ["RL", "Reinforcement Learning", "Learning by trial and error, guided by a reward score"],
      ["SAC", "Soft Actor-Critic", "The RL method we use to train our policy"],
      ["FRC", "FIRST Robotics Competition", "Our league!"],
      ["RPM", "Revolutions Per Minute", "How fast something spins, like our shooter flywheel"],
      ["WPILib", "Worcester Polytechnic Institute Library", "The software library our robot code is built on"],
      ["NT", "NetworkTables", "How the robot and computers share live data"],
      ["OBS", "Observation", "One number the robot “sees”, like its speed or position"],
      ["Ray", "Distance ray", "How far the robot can drive in one direction before hitting something"],
      ["Curriculum", "Training plan", "Practicing easy versions first, then harder ones"],
    ];
    const head = (t) => ({ text: t, options: { bold: true, color: HEX.dk1, fill: { color: HEX.accent1 }, fontSize: 14 } });
    const rows = [[head("Term"), head("Stands for"), head("What it means")]].concat(g.map(([a, b, c], i) => [
      { text: a, options: { bold: true, color: HEX.accent1, fill: { color: i % 2 ? HEX.dk1 : HEX.dk2 }, fontSize: 15 } },
      { text: b, options: { bold: true, color: HEX.lt1, fill: { color: i % 2 ? HEX.dk1 : HEX.dk2 }, fontSize: 15 } },
      { text: c, options: { color: HEX.lt2, fill: { color: i % 2 ? HEX.dk1 : HEX.dk2 }, fontSize: 15 } },
    ]));
    s.addTable(rows, { x: M, y: 1.45, w: W - 2 * M, colW: [1.3, 3.9, W - 2 * M - 5.2], rowH: 0.45,
      border: { type: "none" }, valign: "middle", margin: 0.08, objectName: "glossary table" });
  }
  s.addNotes("Leave this up during questions.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
