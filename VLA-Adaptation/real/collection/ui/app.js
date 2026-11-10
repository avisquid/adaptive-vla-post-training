"use strict";

const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pad = (value) => String(value).padStart(2, "0");
let snapshot = null;
let connected = false;
let pending = false;
let activeView = "collect";
let toastTimer;
let serverOffset = 0;
const clientID = eventID();
const live = () => snapshot?.mode === "collection";
const policyDriving = () => live() && snapshot.session.mode === "recording" && snapshot.session.controller === "policy";
const isRollout = () => live() && snapshot.session.collection_mode === "rollout";
const rolloutPaused = () => isRollout() && snapshot.session.mode === "recording" && snapshot.session.controller === "paused";
const humanDrivingRollout = () => isRollout() && snapshot.session.mode === "recording" && snapshot.session.controller === "human";

function renderPolicyStatus() {
  const pill = $("policy-status");
  const rollout = live() ? snapshot.rollout : null;
  pill.hidden = !rollout;
  if (!rollout) return;
  const {session} = snapshot;
  let text = `Policy ${rollout.checkpoint} · ready`, kind = "";
  if (rollout.status === "error") { text = "Policy unavailable"; kind = "error"; }
  else if (policyDriving()) {
    const latency = rollout.last_planning_s == null ? "" : ` · plan ${rollout.last_planning_s.toFixed(2)} s`;
    // model_prompt is what the worker reports the last plan was conditioned on.
    const used = rollout.model_prompt ? ` · model: “${rollout.model_prompt}”` : "";
    const next = rollout.prompt && rollout.prompt !== rollout.model_prompt ? ` · next plan: “${rollout.prompt}”` : "";
    text = `Policy driving${used}${next} · decision ${rollout.decisions}${rollout.status === "planning" ? " · planning…" : ""}${latency}`;
    kind = "driving";
  } else if (rolloutPaused()) {
    text = `Paused at frame ${session.frames} · T teleop · P resume policy${rollout.prompt ? ` with “${rollout.prompt}”` : ""}`; kind = "human";
  } else if (isRollout() && session.takeover_frame != null && ["recording", "review"].includes(session.mode)) {
    text = `Human driving · ${session.handoffs.length} handoff${session.handoffs.length === 1 ? "" : "s"}`; kind = "human";
  }
  pill.textContent = text;
  pill.className = `policy-pill ${kind}`;
}

function renderPromptSelect(rollout, busy) {
  const select = $("prompt-select");
  const options = rollout?.prompt_options || {};
  $("prompt-control").hidden = !rollout;
  if (!rollout) return;
  const edge = key => /^\d+_\d+$/.test(key) ? key.replace("_", " → ") + " · "
    : key === "default" ? "Default · " : key.replaceAll("_", " ").replace(/^./, c => c.toUpperCase()) + " · ";
  const entries = [["auto", rollout.edge_prompts ? "Auto · follow the route edge" : `Auto · ${options.default}`],
                   ...Object.entries(options).map(([key, text]) => [key, edge(key) + text])];
  const signature = JSON.stringify(entries);
  if (select.dataset.signature !== signature) {
    select.innerHTML = entries.map(([key, label]) => `<option value="${escapeHTML(key)}">${escapeHTML(label)}</option>`).join("");
    select.dataset.signature = signature;
  }
  if (document.activeElement !== select) select.value = rollout.prompt_choice || "auto";
  select.disabled = !connected || busy || !["setup", "recording", "review"].includes(snapshot.session.mode);
}

function renderRouteSelect(busy) {
  const select = $("route-select");
  const {session, task} = snapshot;
  $("route-control").hidden = session.mode !== "setup";
  const entries = task.routes.map((route, i) => {
    const start = task.states[route.states[0]];
    const full = !snapshot.dry_run && session.counts[i] >= route.quota;
    return [i, `Start ${route.states[0]} · ${splitLabel(start).name} · ${route.states.join(" → ")}${full ? " (quota done)" : ""}`, full];
  });
  const signature = JSON.stringify(entries);
  if (select.dataset.signature !== signature) {
    select.innerHTML = entries.map(([i, label, full]) => `<option value="${i}"${full ? " disabled" : ""}>${escapeHTML(label)}</option>`).join("");
    select.dataset.signature = signature;
  }
  if (document.activeElement !== select) select.value = String(session.route_index);
  select.disabled = !connected || busy || session.mode !== "setup";
}

function splitLabel(state) {
  const match = state.label.match(/^\[([^\]]+)\]\s*(.*)$/);
  const name = match ? match[1].replace(/\s+state$/i, "") : `State ${state.id}`;
  return {name: name.charAt(0).toUpperCase() + name.slice(1), description: match ? match[2] : state.label};
}

function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}

function setConnection(value) {
  connected = value;
  $("connection").innerHTML = `<span class="tiny-dot ${value ? "green" : "orange"}"></span>${live() ? "Collector" : "Guide"} ${value ? "connected" : "offline"}`;
  updateControls();
}

function imageMarkup(state) {
  const label = splitLabel(state);
  if (state.image_url) {
    return `<button class="image-area" data-image="${state.id}" aria-label="Enlarge reference for state ${state.id}"><img src="${escapeHTML(state.image_url)}" alt="${escapeHTML(state.label)}" loading="lazy"></button>`;
  }
  return `<div class="image-area"><div class="image-placeholder"><span class="placeholder-number">${pad(state.id)}</span><span class="placeholder-label">${state.image_status === "unsupported" ? "USE A PNG OR JPEG" : "REFERENCE NOT SET"}</span></div></div>`;
}

function render() {
  const {task, session, events} = snapshot;
  const route = task.routes[session.route_index];
  const ids = route.states;
  const total = session.counts.reduce((a, b) => a + b, 0);
  const quota = task.routes.reduce((sum, item) => sum + item.quota, 0);
  const mode = session.mode;
  const isReview = mode === "review";
  const complete = mode === "complete";
  const index = session.segment_index;
  const destination = task.states[ids[index + 1]];
  const label = splitLabel(destination);
  document.title = `${mode === "recording" ? `Goal ${destination.id} · ` : ""}${task.name} · Collection Studio`;
  $("workspace-name").textContent = task.name;
  $("page-title").innerHTML = `${escapeHTML(task.name)} collection<span class="title-dot">.</span>`;
  $("sequence-count").textContent = pad(task.routes.length);
  $("route-list").innerHTML = task.routes.map((r, i) => {
    const done = session.counts[i] >= r.quota;
    return `<div class="route-item ${i === session.route_index ? "current" : ""} ${done ? "done" : ""}" ${i === session.route_index ? 'aria-current="step"' : ""}><span class="route-number">${done ? "✓" : pad(i + 1)}</span><div class="route-path">${r.states.join(" → ")}<span class="route-meta">${session.counts[i]} / ${r.quota} ${live() ? "saved" : "confirmed"}</span></div>${i === session.route_index ? '<span class="route-trailing">↗</span>' : ""}</div>`;
  }).join("");
  $("total-count").textContent = total;
  $("total-quota").textContent = `/ ${quota}`;
  $("current-number").textContent = pad(session.route_index + 1);
  $("route-total").textContent = `/ ${pad(task.routes.length)}`;
  $("current-quota").textContent = `${session.counts[session.route_index]} of ${route.quota} demos`;
  $("marked-count").textContent = `${session.goals.length} of ${ids.length - 1} goals marked`;
  $("sequence-title").innerHTML = `Sequence ${pad(session.route_index + 1)} <span>${ids.join(" → ")}</span>`;
  const modeLabels = {setup:"Setup", recording:"Following goals", review:"Review", interrupted:"Interrupted", complete:"Quota complete", resetting:"Resetting arm", saving:"Saving dataset"};
  $("mode-badge").textContent = modeLabels[mode];
  $("mode-badge").className = `mode-badge ${mode === "recording" ? "recording" : ""}`;
  $("state-strip").innerHTML = ids.map((id, i) => {
    const state = task.states[id];
    const copy = splitLabel(state);
    const reached = i > 0 && i <= session.goals.length;
    const next = !isReview && !complete && i === index + 1;
    const status = isReview || complete ? "Reached" : reached ? "Reached" : i === 0 ? "Start here" : next ? "Next goal" : "Upcoming";
    const arrow = i ? `<div class="card-arrow ${next ? "active" : ""}" aria-hidden="true">→</div>` : "";
    return `${arrow}<article class="state-card ${next ? "goal" : reached || complete ? "reached" : ""}"><div class="card-status"><span>${status}</span><span>${reached || complete ? "✓" : `S${pad(id)}`}</span></div>${imageMarkup(state)}<div class="card-copy"><h3>${escapeHTML(copy.name)}</h3><p>${escapeHTML(copy.description)}</p></div></article>`;
  }).join("");
  $("goal-caption").textContent = complete ? "ALL QUOTAS CONFIRMED" : isReview ? "SEQUENCE REACHED" : "NEXT GOAL";
  $("goal-title").textContent = complete ? "Every sequence is complete." : isReview ? "Ready to review your demonstration." : `State ${destination.id} · ${label.name}`;
  $("goal-description").textContent = complete ? "Export the guide log to keep your session notes." : isReview ? "Save the demo in the collector, then confirm it here." : label.description;
  $("active-edge").innerHTML = isReview || complete ? "✓" : `${ids[index]} <span>→</span> ${ids[index + 1]}`;
  const hints = {setup:"Prepare the first state, then begin your attempt.",recording:"Press G when you reach the highlighted goal.",review:"All goals marked. Confirm after saving in the collector.",interrupted:"Attempt interrupted. Review the error before retrying.",complete:"All sequences have reached their confirmed-demo quotas.",resetting:"Moving to the starting state's joint pose.",saving:"Saving retained transitions and goal annotations."};
  $("sequence-hint").textContent = hints[mode];
  document.querySelector(".sequence-footer .mono").textContent = ["S", ...ids.slice(1).map(() => "G"), "E"].join(" → ");
  const controls = {
    setup:["↗", "Ready to begin", "Arrange the scene to match the starting state. Start recording in the collector, then begin the guide."],
    recording:["◎", `Go to state ${destination.id}`, "Use the SpaceMouse to reach the highlighted goal. Mark your arrival here, even if the robot is stationary."],
    review:["✓", "Sequence reached", "Review and save this demo in the collector. Confirm below to count it toward this sequence."],
    interrupted:["!", "Attempt interrupted", "The guide restarted during this attempt. Check the collector, then retry the guide. No demo has been counted."],
    complete:["✓", "Collection complete", "All sequence quotas are confirmed. Your progress and goal log are saved on this computer."],
    resetting:["↻", "Resetting arm", "Moving to the selected start state's measured qpos. Recording is disabled."],
    saving:["↓", "Saving episode", "Writing the complete route and its annotations. The arm holds its pose."],
  }[mode];
  $("control-icon").textContent = controls[0];
  $("control-title").textContent = controls[1];
  $("control-description").textContent = controls[2];
  $("attempt-number").textContent = `${pad(Math.min(session.counts[session.route_index] + 1, route.quota))} / ${route.quota}`;
  $("save-confirmation").hidden = !isReview;
  $("saved-checkbox").checked = false;
  const readyImages = task.states.filter(s => s.image_url).length;
  const readyPoses = task.states.filter(s => s.qpos_ready).length;
  $("setup-count").textContent = task.states.filter(s => !s.image_url || !s.qpos_ready).length;
  $("readiness").innerHTML = `<div class="readiness-row"><span>▧</span> Reference images <strong>${readyImages} / ${task.states.length} ready</strong></div><div class="readiness-row"><span>⌘</span> Reset poses <strong>${readyPoses} / ${task.states.length} captured</strong></div>`;
  $("recent-activity").innerHTML = activityMarkup(events.slice(0, 3));
  $("full-activity").innerHTML = activityMarkup(events, true);
  $("config-path").textContent = task.config;
  $("state-library").innerHTML = task.states.map(state => {
    const copy = splitLabel(state);
    return `<article class="panel library-card"><div class="library-label"><span>STATE ${pad(state.id)}</span><span>${state.image_url ? "Reference ready" : "Reference needed"}</span></div>${imageMarkup(state)}<h3>${escapeHTML(copy.name)}</h3><p>${escapeHTML(copy.description)}</p><div class="library-label"><span>REFERENCE PATH</span></div><code>${escapeHTML(state.image || "Set image in task YAML")}</code><div class="library-label"><span>RESET QPOS · RADIANS</span><span>${state.qpos_ready ? "Captured" : "Not captured"}</span></div><code>${state.qpos_ready ? state.qpos.map(n => n.toFixed(3)).join(", ") : "Capture offline before robot resets"}</code></article>`;
  }).join("");
  if (live()) renderCollection();
  updateControls();
  updateClock();
  if (mode === "recording" && activeView === "collect") {
    requestAnimationFrame(() => document.querySelector(".state-card.goal")?.scrollIntoView({block:"nearest", inline:"center"}));
  }
}

function activityMarkup(events, date = false) {
  if (!events.length) return '<div class="empty-activity"><span>◷</span><div>Your first attempt starts here.<p>Goal markers and saved-demo confirmations will appear as you go.</p></div></div>';
  const verbs = {start:"Guide attempt started",goal:"Goal reached",undo:"Last goal undone",discard:"Guide attempt discarded",accept:"Saved demo confirmed"};
  if (live()) Object.assign(verbs, {start:"Recording started", discard:"Attempt discarded", accept:"Save requested", reset:"Reset to start", teleop:"Setup teleop toggled", rollout:"Policy rollout started", pause:"Paused", prompt:"Prompt set", route:"Start state chosen"});
  const symbols = {start:"↗",goal:"✓",undo:"↶",discard:"↻",accept:"✓",rollout:"▶",pause:"⏸",teleop:"✋",prompt:"✎"};
  return events.map(event => {
    const stamp = new Date(event.at * 1000);
    const time = date ? stamp.toLocaleString([], {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit",second:"2-digit"}) : stamp.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
    return `<div class="activity-row"><span class="activity-symbol">${symbols[event.action] || "·"}</span><span class="activity-label">${verbs[event.action] || escapeHTML(event.action)}${event.edge ? ` · ${event.edge.join(" → ")}` : ` · Sequence ${pad(event.route_index + 1)}`}</span><time datetime="${escapeHTML(event.at_utc)}">${escapeHTML(time)}</time></div>`;
  }).join("");
}

function updateControls() {
  if (!snapshot) { $("primary-action").disabled = true; return; }
  const {session, task} = snapshot;
  const route = task.routes[session.route_index];
  const mode = session.mode;
  const primary = {
    setup:["Begin attempt", "S", "start"],
    recording:[`Reached state ${route.states[session.segment_index + 1]}`, "G", "goal"],
    review:["Confirm saved demo", "E", "accept"],
    interrupted:["Retry guide attempt", "R", "discard"],
    complete:["All quotas complete", "✓", "none"],
    resetting:["Resetting…", "H", "none"],
    saving:["Saving…", "E", "none"],
  }[mode];
  if (live() && mode === "setup") primary[0] = "Start recording";
  if (live() && mode === "review") primary[0] = snapshot.dry_run ? "Finish practice" : "Save demonstration";
  if (live() && mode === "review" && snapshot.save_reset_state_id != null)
    primary[0] = `${snapshot.dry_run ? "Finish" : "Save"} & reset to state ${snapshot.save_reset_state_id}`;
  if (live() && mode === "interrupted") primary[0] = "Discard & reset";
  $("primary-action").innerHTML = `<span>${pending ? "Updating…" : primary[0]}</span><kbd>${primary[1]}</kbd>`;
  $("primary-action").dataset.action = primary[2];
  const busy = pending || snapshot.command_pending || snapshot.fatal;
  $("primary-action").disabled = !connected || busy || ["complete", "resetting", "saving"].includes(mode) || (live() ? mode === "setup" && !session.reset_ready : mode === "review" && !$("saved-checkbox").checked);
  $("undo-action").disabled = !connected || busy || !["recording", "review"].includes(mode) || !session.goals.length;
  $("retry-action").disabled = !connected || busy || !["recording", "review", "interrupted"].includes(mode);
  if (live()) {
    $("reset-action").hidden = mode !== "setup";
    $("retry-action").hidden = !["recording", "review"].includes(mode);
    $("retry-action").innerHTML = 'Discard & reset <kbd>R</kbd>';
    $("reset-action").disabled = !connected || busy || mode !== "setup";
    // In a rollout recording the policy, a pause and the SpaceMouse take turns in any order.
    const canTeleop = isRollout() && mode === "recording" && session.controller !== "human";
    const canHandToPolicy = isRollout() && mode === "recording" && session.controller !== "policy";
    $("teleop-action").disabled = !connected || busy || (mode !== "setup" && !canTeleop);
    $("teleop-action").hidden = mode !== "setup" && !canTeleop;
    $("simulate-action").disabled = !connected || busy || mode !== "recording";
    $("teleop-action").textContent = rolloutPaused() ? "Continue with teleop (T)" : policyDriving() ? "Take over with SpaceMouse (T)" : session.teleop_enabled ? "Disable setup teleop (T)" : "Enable setup teleop (T)";
    const rollout = snapshot.rollout;
    $("rollout-action").hidden = !rollout || !(mode === "setup" || canHandToPolicy);
    $("rollout-action").innerHTML = rolloutPaused() ? 'Resume policy <kbd>P</kbd>' : humanDrivingRollout() ? 'Hand to policy <kbd>P</kbd>' : 'Run policy <kbd>P</kbd>';
    $("rollout-action").disabled = !connected || busy || !rollout?.available || (canHandToPolicy ? false : mode !== "setup" || !session.reset_ready);
    $("takeover-action").hidden = !(policyDriving() || humanDrivingRollout());
    $("takeover-action").disabled = !connected || pending || snapshot.fatal;
    if (policyDriving()) $("simulate-action").disabled = true;
    renderPromptSelect(rollout, busy);
    renderRouteSelect(busy);
  }
  renderPolicyStatus();
}

function renderCollection() {
  const {session,simulation,dry_run} = snapshot;
  const resetAfterSave = snapshot.auto_reset_after_save;
  const nextStart = snapshot.save_reset_state_id;
  const saveStatus = snapshot.save_status;
  document.querySelector(".guide-badge").textContent = simulation ? "Simulation" : dry_run ? "Hardware practice" : snapshot.rollout ? "Rollout collection" : "Single-FR3 collection";
  document.querySelector(".guide-notice p").textContent = simulation ? "Hardware-free simulation. H prepares the start, S begins, M simulates movement, G marks a goal, E finishes. No robot is connected." : "H resets to the starting state's qpos. T enables setup teleop. S starts recording; G marks each goal. " + (resetAfterSave ? "E resets to the next attempt's start, then finishes saving while you arrange the scene. Clear the arm's path before E." : "E saves the route. Use H to reset for the next attempt.");
  if (!simulation && snapshot.task.agilex_qpos != null)
    document.querySelector(".guide-notice p").textContent += " Resets position the camera arm first; clear both arms' paths.";
  document.querySelector(".stat>span").textContent = "ACCEPTED DEMOS";
  document.querySelector(".stat-note").textContent = dry_run ? "practice · no data written" : "saved in dataset";
  $("save-confirmation").hidden = true;
  $("robot-controls").hidden = false;
  $("simulate-action").hidden = !simulation;
  $("reset-action").innerHTML = `Reset to state ${snapshot.task.routes[session.route_index].states[0]} <kbd>H</kbd>`;
  if (session.mode === "setup") {
    $("control-title").textContent = session.reset_ready ? "Ready to record" : "Prepare the start";
    $("control-description").textContent = session.reset_ready ? "Arrange the scene, optionally enable setup teleop, then start recording." : "Clear the path, reset the arm to the starting state's captured pose, and arrange the scene.";
    if (dry_run && !simulation && !session.reset_ready) {
      $("control-title").textContent = "Practice teleoperation";
      $("control-description").textContent = "Press T to control the arm from its current pose. A captured qpos is needed only when you reset to practice a route.";
    }
  }
  if (session.mode === "recording") $("control-description").textContent = "Use the SpaceMouse to reach the highlighted state. G marks arrival even while stationary; only taken transitions are recorded.";
  if (snapshot.rollout && session.mode === "setup" && session.reset_ready)
    $("control-description").textContent += " P runs the policy instead; S records with the SpaceMouse, and the policy can still take over later.";
  if (policyDriving()) {
    $("control-title").textContent = "Policy driving";
    $("control-description").textContent = (snapshot.rollout.prompt ? `Instruction: “${snapshot.rollout.prompt}”. ` : "") + "The policy controls the arm. Press T to take over with the SpaceMouse, or Space (or the orange button) to pause. G still marks each goal" + (snapshot.rollout.prompt_choice === "auto" ? " and switches the instruction to the next edge." : "; the chosen prompt stays until you change it.");
  } else if (rolloutPaused()) {
    $("control-title").textContent = "Paused";
    $("control-description").textContent = "The arm holds its pose. Press T to continue with the SpaceMouse, or choose a prompt and press P to resume the policy. G still marks a goal.";
  } else if (isRollout() && session.mode === "recording") {
    $("control-title").textContent = "You have control";
    $("control-description").textContent = "Drive with the SpaceMouse; these frames are labelled as human interventions. P hands the arm to the policy, Space pauses. G marks each goal.";
  }
  if (session.mode === "review") {
    $("control-description").textContent = dry_run ? "All goals reached. Finish this practice attempt; no data or quotas are changed." : "All goals reached. E saves this complete trajectory with its edge annotations.";
    if (nextStart != null) $("control-description").textContent += ` The arm resets to state ${nextStart} before video encoding, with the configured pose noise. Clear the path before pressing E.`;
    $("goal-description").textContent = "Review the marked goals, then save here.";
  }
  if (session.mode === "saving")
    $("control-description").textContent = saveStatus?.reset_complete
      ? `Reset to state ${nextStart} complete. Arrange the scene while the episode saves. S becomes available when saving finishes.`
      : "Saving the completed episode. Controls become available when saving finishes.";
  if (session.mode === "resetting")
    $("control-description").textContent = `Moving to state ${nextStart ?? snapshot.task.routes[session.route_index].states[0]} with the configured pose noise. Recording is off.`;
  if (session.mode === "resetting" && snapshot.task.agilex_qpos != null)
    $("control-description").textContent = "Positioning the camera arm, then resetting the FR3. Recording is off.";
  if (session.mode === "interrupted") $("control-description").textContent = snapshot.fatal ? "Inspect the error and restart the collector. The draft remains on disk." : "The attempt stopped. Discard and retry after reconnecting.";
  if (session.mode === "complete") $("goal-description").textContent = "All configured route quotas are saved in the dataset.";
  document.querySelector('a[href="/api/export"]').textContent = "↓ Export session log";
  document.querySelector(".log-explainer").textContent = "Goal events include retained-frame boundaries and original timestamps. Accepted episodes carry their annotations in the dataset.";
  document.querySelector(".readiness-note").textContent = "Reset poses and reference images come from the shared task config.";
  document.querySelector("#help-dialog > p").textContent = "This page controls the single-arm collection session. Keep it focused for keyboard shortcuts.";
  document.querySelector(".help-steps").innerHTML = '<li>Clear the reset path. Press <kbd>H</kbd> to move to the starting state’s qpos; arrange the scene.</li><li>Optionally use <kbd>T</kbd> for setup teleop, then <kbd>S</kbd> to record.</li><li>Use the SpaceMouse. Press <kbd>G</kbd> at each goal, including while stationary.</li><li>Review the complete route, then <kbd>E</kbd> to save. <kbd>U</kbd> corrects the last goal and <kbd>R</kbd> discards an attempt.</li>';
  document.querySelector(".shortcut-grid").innerHTML = (snapshot.rollout ? '<span><kbd>P</kbd> Run / resume policy</span><span><kbd>Space</kbd> Pause</span><span><kbd>T</kbd> Take over with SpaceMouse</span>' : '') + '<span><kbd>U</kbd> Undo last goal</span><span><kbd>R</kbd> Discard & reset</span><span><kbd>?</kbd> Open help</span><span><kbd>Esc</kbd> Close dialog</span>';
  document.querySelector(".help-note").textContent = (resetAfterSave ? "E preserves the episode, resets to the next attempt's first state, then finishes saving while you arrange the scene. Clear the arm's path before E. " : "Use H to reset after saving. ") + "Discarding resets to the same route's start. Every reset opens the gripper. Closing this page during teleop interrupts the attempt. Keep the physical stop accessible.";
  $("sequence-hint").textContent = {setup:"Reset to the first state, arrange the scene, then start recording.", recording:"Press G at the highlighted goal, even while stationary.", review:"All goals marked. Save this complete route here.", interrupted:"Review the error before retrying.", complete:"All route quotas are saved.", resetting:"Moving to the starting state’s qpos.", saving:"Saving transitions and annotations."}[session.mode];
  if (session.mode === "setup" && session.reset_ready)
    $("sequence-hint").textContent = snapshot.rollout ? "Arm reset complete. Arrange the scene, then press P to run the policy or S to record a demo." : "Arm reset complete. Arrange the scene, then press S to record.";
  if (policyDriving()) $("sequence-hint").textContent = "Policy driving. T takes over, Space pauses; press G at each goal.";
  if (humanDrivingRollout()) $("sequence-hint").textContent = "You drive. P hands to the policy, Space pauses; press G at each goal.";
  if (rolloutPaused()) $("sequence-hint").textContent = "Paused. T continues with teleop; P resumes the policy with the selected prompt.";
  if (session.mode === "review" && nextStart != null)
    $("sequence-hint").textContent = `E resets to state ${nextStart}, then finishes saving. Clear the arm's path first.`;
  if (session.mode === "saving" && saveStatus?.reset_complete)
    $("sequence-hint").textContent = `Arm at state ${nextStart}. Arrange the scene while saving finishes.`;
  document.querySelector(".activity-panel .quiet").textContent = "Collection events";
  document.querySelector(".controls-note p").textContent = "One SpaceMouse controls the arm and gripper. Stationary ticks add no transitions.";
  document.querySelector("#retry-dialog p").textContent = "Discard this attempt, open the gripper, and reset to the same sequence's starting state. Clear the arms' paths before confirming. Staged data remains available for inspection.";
  $("confirm-retry").textContent = "Discard & reset";
  $("preview-kind").textContent = ["saving", "resetting"].includes(session.mode) ? "Camera preview paused" : simulation ? "Synthetic camera preview" : "Live camera preview";
}

function updateClock() {
  if (!snapshot) return;
  const {session} = snapshot;
  let elapsed = 0;
  if (session.started_at) {
    const end = session.mode === "review" ? session.goals.at(-1).at : session.mode === "interrupted" ? session.started_at : Date.now() / 1000 + serverOffset;
    elapsed = Math.max(0, Math.floor(end - session.started_at));
  }
  $("elapsed").textContent = `${pad(Math.floor(elapsed / 60))}:${pad(elapsed % 60)}`;
}

function applySnapshot(data) {
  if (snapshot && data.session.revision < snapshot.session.revision) return;
  const changed = !snapshot || data.task.fingerprint !== snapshot.task.fingerprint || data.session.revision !== snapshot.session.revision;
  serverOffset = data.server_time - Date.now() / 1000;
  snapshot = data;
  setConnection(true);
  $("error-banner").hidden = true;
  if (changed) render();
  if (live()) {
    $("marked-count").textContent = `${snapshot.session.frames} taken transitions · ${snapshot.session.goals.length} goals`
      + (isRollout() ? ` · ${snapshot.session.policy_frames} policy / ${snapshot.session.frames - snapshot.session.policy_frames} human` : "");
    renderPolicyStatus();
    if (snapshot.error) { $("error-banner").hidden = false; $("error-banner").textContent = snapshot.error; }
    $("live-previews").hidden = !snapshot.cameras.length;
    const views = $("camera-views");
    if (views.children.length !== snapshot.cameras.length) views.innerHTML = snapshot.cameras.map(name => `<figure><img data-camera="${escapeHTML(name)}" alt="${escapeHTML(name)} camera"><figcaption>${escapeHTML(name)}</figcaption></figure>`).join("");
  }
}

async function refresh() {
  try {
    const response = await fetch(`/api/state?client_id=${clientID}`, {signal:AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("Collection server is unavailable.");
    applySnapshot(await response.json());
  } catch (error) {
    setConnection(false);
    $("error-banner").textContent = live() ? "Connection to the collector lost; reconnecting automatically. Teleop interrupts after the connection timeout." : "Connection to the guide lost. Controls are paused; reconnecting automatically.";
    $("error-banner").hidden = false;
  }
}

function eventID() {
  // randomUUID requires HTTPS; LAN HTTP still supports getRandomValues.
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
}

async function command(action, extra = {}) {
  if (!snapshot || !connected || pending) return;
  if (action === "accept" && !live() && !$("saved-checkbox").checked) {
    toast("Save in the collector, then check the confirmation box first.");
    return;
  }
  pending = true;
  updateControls();
  try {
    const response = await fetch("/api/command", {
      method:"POST", headers:{"Content-Type":"application/json"}, signal:AbortSignal.timeout(7000),
      body:JSON.stringify({action, client_id:clientID, event_id:eventID(), revision:snapshot.session.revision, saved_in_collector:action === "accept", ...extra}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Action failed.");
    const previousRoute = snapshot.session.route_index;
    applySnapshot(data);
    const messages = {start:"Guide started. Follow the highlighted goal.",goal:"Goal marked.",undo:"Last goal undone.",discard:"Ready for a fresh attempt.",accept:"Saved demo confirmed.",prompt:"Prompt applies from the next policy plan.",route:"Start state chosen. Press H to reset there."};
    toast(data.command_pending ? "Request sent to collector…" : data.session.route_index !== previousRoute ? "Sequence quota complete. Prepare the next sequence." : messages[action]);
  } catch (error) {
    toast(error.name === "TimeoutError" ? "No acknowledgement received. Checking saved guide state…" : error.message);
    await refresh();
  } finally {
    pending = false;
    updateControls();
  }
}

function changeView(view) {
  activeView = view;
  document.querySelectorAll(".view").forEach(el => { el.hidden = el.id !== `${view}-view`; });
  document.querySelectorAll(".nav-button").forEach(button => {
    button.classList.toggle("active", button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("page-name").textContent = {collect:"Collect",setup:"Task setup",log:"Session log"}[view];
  $("page-subtitle").textContent = {collect:"A clear path from one state to the next.",setup:"The shared references behind every demonstration.",log:"A record of the path you took."}[view];
}

document.addEventListener("click", event => {
  const viewButton = event.target.closest("[data-view]");
  if (viewButton) changeView(viewButton.dataset.view);
  const closeButton = event.target.closest("[data-close]");
  if (closeButton) $(closeButton.dataset.close).close();
  const imageButton = event.target.closest("[data-image]");
  if (imageButton && snapshot) {
    const state = snapshot.task.states[Number(imageButton.dataset.image)];
    $("expanded-image").src = state.image_url;
    $("expanded-image").alt = state.label;
    $("expanded-title").textContent = `State ${state.id} · ${splitLabel(state).name}`;
    $("expanded-description").textContent = splitLabel(state).description;
    $("image-dialog").showModal();
  }
});
document.addEventListener("error", event => {
  if (event.target.dataset?.camera) return; // The next preview request can recover.
  if (event.target.tagName === "IMG" && event.target !== $("expanded-image")) {
    const parent = event.target.parentElement;
    parent.disabled = true;
    parent.innerHTML = '<div class="image-placeholder"><span class="placeholder-label">IMAGE COULD NOT LOAD</span></div>';
  }
}, true);
$("primary-action").addEventListener("click", () => {
  const action = $("primary-action").dataset.action;
  if (action === "discard") $("retry-dialog").showModal();
  else command(action);
});
$("undo-action").addEventListener("click", () => command("undo"));
$("retry-action").addEventListener("click", () => $("retry-dialog").showModal());
$("confirm-retry").addEventListener("click", () => { $("retry-dialog").close(); command("discard"); });
$("saved-checkbox").addEventListener("change", updateControls);
$("reset-action").addEventListener("click", () => command("reset"));
$("teleop-action").addEventListener("click", () => command("teleop"));
$("simulate-action").addEventListener("click", () => command("simulate_motion"));
$("rollout-action").addEventListener("click", () => command("rollout"));
$("takeover-action").addEventListener("click", () => command("pause"));
// Blur after choosing so Space (pause) and the other shortcuts keep working.
$("prompt-select").addEventListener("change", event => { event.target.blur(); command("prompt", {prompt_choice:event.target.value}); });
$("route-select").addEventListener("change", event => { event.target.blur(); command("route", {route_index:Number(event.target.value)}); });
$("help-button").addEventListener("click", () => $("help-dialog").showModal());
$("about-button").addEventListener("click", () => $("help-dialog").showModal());
document.addEventListener("keydown", event => {
  if (event.repeat || event.ctrlKey || event.altKey || event.metaKey || document.querySelector("dialog[open]") || event.target.closest("input:not([type=checkbox]):not([type=radio]),textarea,select,[contenteditable=true]")) return;
  const key = event.key.toLowerCase();
  if (key === "?") { event.preventDefault(); $("help-dialog").showModal(); return; }
  if (activeView !== "collect" || !snapshot || !connected || pending) return;
  const mode = snapshot.session.mode;
  if (live()) {
    const extra = {h:["reset", "reset-action"],t:["teleop", "teleop-action"],m:["simulate_motion", "simulate-action"],p:["rollout", "rollout-action"]," ":["pause", "takeover-action"]}[key];
    if (extra && !$(extra[1]).disabled && !$(extra[1]).hidden) { event.preventDefault(); command(extra[0]); return; }
  }
  const action = key === "s" && mode === "setup" ? "start" : key === "g" && mode === "recording" ? "goal" : key === "e" && mode === "review" ? "accept" : key === "u" && !$("undo-action").disabled ? "undo" : null;
  if (action) { event.preventDefault(); command(action); }
  if (key === "r" && !$("retry-action").disabled) { event.preventDefault(); $("retry-dialog").showModal(); }
});

async function poll() { await refresh(); setTimeout(poll, live() ? 200 : 1200); }
updateControls();
poll();
setInterval(updateClock, 500);
setInterval(() => { if (live() && connected) document.querySelectorAll('[data-camera]').forEach(img => { img.src = `/api/preview/${encodeURIComponent(img.dataset.camera)}?t=${Date.now()}`; }); }, 200);
