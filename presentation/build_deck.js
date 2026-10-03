// build_deck.js — "Teaching a Robot to Drive Itself" (RL intro, Advanced Programming lesson 1)
//
//   python presentation/prep_media.py     # copy clips + posters into presentation/media
//   node presentation/build_deck.js       # writes presentation/RL_Intro_Lesson1.pptx
//
// Re-run both after the path_randomizer training run produces more videos (Chapter 3).

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
  s.addNotes("Walk around the loop. The policy is a small neural network. Every 20 milliseconds (the same loop timing as our real robot code) it sees, acts, and gets scored. Training means nudging the network so actions that led to more points become more likely.");

  // 7 ─ Words to know
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What is RL?" });
  s.addText("Words to know", { placeholder: "title" });
  {
    const words = [
      { w: "Policy", d: "The robot's brain: a small neural network that turns what it sees into what it does.", Comp: fa.FaBrain, fill: HEX.accent1, ic: HEX.dk1 },
      { w: "Observation", d: "What the robot “sees”: its speed, its position, and where its next waypoint is.", Comp: fa.FaEye, fill: HEX.accent6, ic: HEX.dk1 },
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
    const plus = [
      ["+", "Moving forward along the path"],
      ["+", "Bonus for each waypoint passed"],
      ["−", "Drifting away from the line"],
      ["−", "Jerky back-and-forth wiggles"],
      ["−", "A tiny cost every moment (hurry up!)"],
    ];
    s.addText(plus.map(([sgn, t], i) => ({
      text: `${sgn}  ${t}`,
      options: { color: sgn === "+" ? C.accent4 : C.accent2, breakLine: i < plus.length - 1 } })),
    { x: rx, y: 2.05, w: rw, h: 2.6, isTextBox: true, fontSize: 17, margin: 0, valign: "top", paraSpaceAfter: 9 });
    s.addText("After 12,000 steps the score barely moves. It's already about as good as these points can make it.", {
      x: rx, y: 4.9, w: rw, h: 1.5, isTextBox: true, fontSize: 16, italic: true, color: C.accent1, margin: 0, valign: "top" });
  }
  s.addNotes("Each dot is the score from one test drive recorded during training. Flat and negative while it explores, then a jump between 8,000 and 12,000 steps. The right side is the whole 'reward function' in plain words: that list is all the robot was told.");

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
      { text: "Same robot, same size brain. ", options: { bold: true, color: C.background1 } },
      { text: "Adding obstacles meant it needed about 6× more practice (12,000 → 75,000 steps) before its first full lap." },
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
  s.addText("So far it learned ONE path by heart. Real autonomous routines change every match. New idea: a brand-new random path for every attempt.", { placeholder: "body" });
  s.addNotes("Memorizing one route is like memorizing one test's answers. We want it to actually understand driving to a point.");

  // 16 ─ Random paths
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Chapter 3: Random paths" });
  s.addText("Every attempt is a new puzzle", { placeholder: "title" });
  {
    const lw = 4.6;
    const pts = [
      { Comp: fa.FaRandom, head: "Random every time", t: "3 to 12 waypoints anywhere on the field, each up to 6 m apart." },
      { Comp: fa.FaCompass, head: "Sees where to go next", t: "It sees the direction and distance to its next two waypoints, not a memorized route." },
      { Comp: fa.FaBullseye, head: "New scoring", t: "Points for getting closer than ever before, +100 for reaching a waypoint, +75 for finishing, −75 for a crash." },
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
    const early = manifest.ch3_early, late = manifest.ch3_late;
    // Two stacked rows: video on the left, label on the right.
    const rowW = 5.0, rowH = rowW * 640 / 1440, lx = vx + rowW + 0.3, lw2 = W - M - lx;
    const rows = [
      { key: "ch3_early", y: 1.5, sub: "Just starting out: random moves while it explores." },
      { key: "ch3_late", y: 1.5 + rowH + 0.5, sub: "Latest test drive. Still training on a laptop right now!" },
    ];
    for (const r of rows) {
      const m = manifest[r.key];
      if (m && !(r.key === "ch3_late" && manifest.ch3_early && m.train_step <= manifest.ch3_early.train_step)) {
        await video(s, r.key, vx, r.y, rowW, r.key + " video");
        s.addText([
          { text: `${steps(m.train_step)} steps`, options: { bold: true, fontSize: 20, color: C.accent1, breakLine: true } },
          { text: r.sub, options: { fontSize: 15 } },
        ], { x: lx, y: r.y, w: lw2, h: rowH, isTextBox: true, color: C.background2, valign: "middle", margin: 0, paraSpaceAfter: 6 });
      } else {
        placeholderVideo(s, vx, r.y, rowW, rowH, "Training video coming soon", r.key + " placeholder");
      }
    }
  }
  s.addNotes("Each yellow dot is the current target waypoint; the path is different every attempt. Compare the early clip (lost) with the latest one. This is the policy we're bringing to our real robot code.");

  // 17 ─ Sim to robot
  pres.addSection({ title: "This season" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "This season" });
  s.addText("From the simulator to our robot", { placeholder: "title" });
  {
    const stages = [
      { Comp: fa.FaLaptopCode, fill: HEX.accent1, ic: HEX.dk1, head: "1. Train",
        sub: "Swerve Policy Playground", t: "Python simulator. The robot practices millions of steps." },
      { Comp: fa.FaVial, fill: HEX.accent3, ic: HEX.lt1, head: "2. Test",
        sub: "1507Labs", t: "Our real robot code in the WPILib simulator. Does the policy drive our actual swerve code?" },
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
    ];
    const head = (t) => ({ text: t, options: { bold: true, color: HEX.dk1, fill: { color: HEX.accent1 }, fontSize: 14 } });
    const rows = [[head("Short"), head("Stands for"), head("What it means")]].concat(g.map(([a, b, c], i) => [
      { text: a, options: { bold: true, color: HEX.accent1, fill: { color: i % 2 ? HEX.dk1 : HEX.dk2 }, fontSize: 15 } },
      { text: b, options: { bold: true, color: HEX.lt1, fill: { color: i % 2 ? HEX.dk1 : HEX.dk2 }, fontSize: 15 } },
      { text: c, options: { color: HEX.lt2, fill: { color: i % 2 ? HEX.dk1 : HEX.dk2 }, fontSize: 15 } },
    ]));
    s.addTable(rows, { x: M, y: 1.45, w: W - 2 * M, colW: [1.4, 4.3, W - 2 * M - 5.7], rowH: 0.56,
      border: { type: "none" }, valign: "middle", margin: 0.08, objectName: "glossary table" });
  }
  s.addNotes("Leave this up during questions.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
