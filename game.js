import * as THREE from 'https://unpkg.com/three@0.166.1/build/three.module.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x040509);
scene.fog = new THREE.FogExp2(0x090a0f, 0.045);

const camera = new THREE.PerspectiveCamera(72, window.innerWidth / window.innerHeight, 0.1, 350);
camera.position.set(0, 1.7, 20);

const listener = new THREE.AudioListener();
camera.add(listener);
const audioCtx = listener.context;

const ui = {
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadingFill: document.getElementById('loadingFill'),
  loadingHint: document.getElementById('loadingHint'),
  startOverlay: document.getElementById('startOverlay'),
  deathOverlay: document.getElementById('deathOverlay'),
  winOverlay: document.getElementById('winOverlay'),
  objective: document.getElementById('objective'),
  keys: document.getElementById('keys'),
  stamina: document.getElementById('stamina'),
  ammo: document.getElementById('ammo'),
  sanity: document.getElementById('sanity'),
  health: document.getElementById('health'),
  noise: document.getElementById('noise'),
  daytime: document.getElementById('daytime'),
  inventoryOverlay: document.getElementById('inventoryOverlay'),
  threat: document.getElementById('threat')
};

const game = {
  started: false,
  over: false,
  won: false,
  keysTotal: 3,
  keysFound: 0,
  ammo: 0,
  stamina: 100,
  sanity: 100,
  health: 3,
  noiseLevel: 0,
  daySeconds: 0,
  dayIndex: 1,
  running: false,
  crouching: false,
  holdBreath: false,
  inventoryOpen: false,
  gateUnlocked: false,
  loadingDone: false
};

const player = {
  pos: new THREE.Vector3(0, 1.7, 20),
  yaw: Math.PI,
  pitch: 0,
  move: new THREE.Vector2(0, 0)
};

const ghost = {
  root: null,
  pos: new THREE.Vector3(0, 0.9, -24),
  speed: 1.35,
  rageSpeed: 2.25,
  rage: false,
  cooldown: 0,
  stunned: 0,
  footPhase: 0,
  howlTimer: 0
};

const world = {
  walls: [],
  props: [],
  closets: [],
  keys: [],
  ammoPickups: [],
  stairs: [],
  hiddenRoomCenter: new THREE.Vector3(23, 0, 23),
  boundsMin: new THREE.Vector3(-31, 0, -31),
  boundsMax: new THREE.Vector3(31, 0, 31)
};

const clock = new THREE.Clock();

function makeCanvasTexture(size, paint) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  paint(ctx, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

function createProceduralTextures() {
  const floorTex = makeCanvasTexture(512, (ctx, s) => {
    ctx.fillStyle = '#2f261f';
    ctx.fillRect(0, 0, s, s);
    for (let y = 0; y < s; y += 32) {
      const jitter = (Math.random() * 4) | 0;
      ctx.fillStyle = y % 64 === 0 ? '#3b3028' : '#342a23';
      ctx.fillRect(0, y, s, 30 + jitter);
      ctx.strokeStyle = 'rgba(20,12,8,0.35)';
      ctx.strokeRect(0, y, s, 30 + jitter);
    }
    for (let i = 0; i < 2400; i++) {
      ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.08})`;
      ctx.fillRect(Math.random() * s, Math.random() * s, 1, 1);
    }
  });
  floorTex.repeat.set(16, 16);

  const wallTex = makeCanvasTexture(512, (ctx, s) => {
    ctx.fillStyle = '#2b2a31';
    ctx.fillRect(0, 0, s, s);
    const brickW = 64;
    const brickH = 28;
    for (let y = 0; y < s; y += brickH) {
      for (let x = 0; x < s; x += brickW) {
        const ox = (Math.floor(y / brickH) % 2) * (brickW / 2);
        const bx = x + ox;
        const shade = 45 + ((x + y) % 40);
        ctx.fillStyle = `rgb(${shade},${shade-8},${shade+4})`;
        ctx.fillRect(bx + 2, y + 2, brickW - 4, brickH - 4);
      }
    }
    ctx.strokeStyle = 'rgba(12,12,16,0.5)';
    for (let y = 0; y < s; y += brickH) ctx.strokeRect(0, y, s, brickH);
  });
  wallTex.repeat.set(10, 3);

  const ceilingTex = makeCanvasTexture(256, (ctx, s) => {
    ctx.fillStyle = '#10141c';
    ctx.fillRect(0, 0, s, s);
    for (let i = 0; i < 1800; i++) {
      const c = 20 + ((Math.random() * 30) | 0);
      ctx.fillStyle = `rgb(${c},${c},${c + 8})`;
      ctx.fillRect(Math.random() * s, Math.random() * s, 1, 1);
    }
  });
  ceilingTex.repeat.set(8, 8);

  return { floorTex, wallTex, ceilingTex };
}


function tone(freq, duration, type = 'sine', volume = 0.07, glide = 0) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  if (glide) osc.frequency.linearRampToValueAtTime(freq + glide, audioCtx.currentTime + duration);
  gain.gain.setValueAtTime(volume, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + duration);
}

function noiseBurst(duration = 0.3, volume = 0.02) {
  const sr = audioCtx.sampleRate;
  const buffer = audioCtx.createBuffer(1, sr * duration, sr);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length);
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  const gain = audioCtx.createGain();
  gain.gain.value = volume;
  src.connect(gain);
  gain.connect(audioCtx.destination);
  src.start();
}

function setObjective(text) { ui.objective.textContent = text; }

function updateHUD() {
  ui.keys.textContent = `${game.keysFound} / ${game.keysTotal}`;
  ui.stamina.textContent = `${Math.round(game.stamina)}%`;
  ui.ammo.textContent = `${game.ammo}`;
  ui.sanity.textContent = `${Math.round(game.sanity)}%`;
  ui.health.textContent = `${game.health} / 3`;
  ui.noise.textContent = game.noiseLevel > 0.66 ? 'High' : game.noiseLevel > 0.33 ? 'Medium' : 'Low';

  const total = game.daySeconds % (20 * 60);
  const phaseHour = Math.floor((total / (20 * 60)) * 24);
  const hh = String(phaseHour).padStart(2, '0');
  ui.daytime.textContent = `Day ${game.dayIndex} · ${hh}:00`;

  ui.threat.textContent = ghost.stunned > 0 ? 'Stunned' : ghost.rage ? 'Hunting' : 'Searching';
}

function addWall(x, z, w, h, d, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  mesh.position.set(x, h / 2, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);
  world.walls.push(new THREE.Box3().setFromObject(mesh));
  world.props.push(mesh);
}

function makeBed(x, z, rot = 0) {
  const g = new THREE.Group();
  const frame = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.45, 4), new THREE.MeshStandardMaterial({ color: 0x3f2e2e }));
  const mattress = new THREE.Mesh(new THREE.BoxGeometry(2, 0.4, 3.7), new THREE.MeshStandardMaterial({ color: 0xccc3b8 }));
  const head = new THREE.Mesh(new THREE.BoxGeometry(2.2, 1.1, 0.2), new THREE.MeshStandardMaterial({ color: 0x2a1a1a }));
  mattress.position.y = 0.42;
  head.position.set(0, 0.8, -1.9);
  g.add(frame, mattress, head);
  g.position.set(x, 0, z);
  g.rotation.y = rot;
  scene.add(g);
  world.walls.push(new THREE.Box3().setFromObject(g));
  world.props.push(g);
}

function makeDrawer(x, z) {
  const g = new THREE.Group();
  const shell = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.2, 0.7), new THREE.MeshStandardMaterial({ color: 0x4d382d }));
  const h1 = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.25, 0.04), new THREE.MeshStandardMaterial({ color: 0x7c5a4f }));
  const h2 = h1.clone();
  const h3 = h1.clone();
  h1.position.set(0, 0.4, 0.36); h2.position.set(0, 0, 0.36); h3.position.set(0, -0.4, 0.36);
  g.add(shell, h1, h2, h3);
  g.position.set(x, 0.6, z);
  scene.add(g);
  world.walls.push(new THREE.Box3().setFromObject(g));
  world.props.push(g);
}

function makeCloset(x, z) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.8, 2.8, 0.9), new THREE.MeshStandardMaterial({ color: 0x2b2626 }));
  const leftDoor = new THREE.Mesh(new THREE.BoxGeometry(0.85, 2.65, 0.05), new THREE.MeshStandardMaterial({ color: 0x463737 }));
  const rightDoor = leftDoor.clone();
  leftDoor.position.set(-0.44, 0, 0.48);
  rightDoor.position.set(0.44, 0, 0.48);
  g.add(body, leftDoor, rightDoor);
  g.position.set(x, 1.4, z);
  scene.add(g);
  world.walls.push(new THREE.Box3().setFromObject(g));
  world.closets.push(g);
  world.props.push(g);
}

function makeStairs(x, z, steps = 8, rot = 0) {
  const group = new THREE.Group();
  for (let i = 0; i < steps; i++) {
    const s = new THREE.Mesh(new THREE.BoxGeometry(2.8, 0.23, 0.9), new THREE.MeshStandardMaterial({ color: 0x2e2c35 }));
    s.position.set(0, i * 0.23, i * 0.85);
    group.add(s);
  }
  group.position.set(x, 0.1, z);
  group.rotation.y = rot;
  scene.add(group);
  world.stairs.push(group);
  world.walls.push(new THREE.Box3().setFromObject(group));
  world.props.push(group);
}

function createKeyMesh() {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.05, 10, 22), new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.9, roughness: 0.2 }));
  const stem = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.45, 0.08), ring.material);
  stem.position.y = -0.31;
  const tooth1 = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.08, 0.08), ring.material);
  const tooth2 = tooth1.clone();
  tooth1.position.set(0.08, -0.5, 0);
  tooth2.position.set(-0.02, -0.56, 0);
  g.add(ring, stem, tooth1, tooth2);
  return g;
}

function createGunPickup() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.12, 0.2), new THREE.MeshStandardMaterial({ color: 0x111827, metalness: 0.8, roughness: 0.3 }));
  const handle = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.3, 0.16), body.material);
  handle.position.set(-0.15, -0.16, 0);
  g.add(body, handle);
  return g;
}

function setupWorld() {
  const tex = createProceduralTextures();

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(70, 70), new THREE.MeshStandardMaterial({ map: tex.floorTex, roughness: 0.9 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(70, 70), new THREE.MeshStandardMaterial({ map: tex.ceilingTex, roughness: 1 }));
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = 5;
  scene.add(ceiling);

  const wallMat = new THREE.MeshStandardMaterial({ map: tex.wallTex, roughness: 0.95 });
  addWall(0, -35, 70, 5, 1, wallMat); addWall(0, 35, 70, 5, 1, wallMat);
  addWall(-35, 0, 1, 5, 70, wallMat); addWall(35, 0, 1, 5, 70, wallMat);

  const partitions = [
    [-20, 0, 1, 5, 48], [20, 0, 1, 5, 48], [0, -14, 40, 5, 1], [0, 14, 40, 5, 1],
    [-10, 24, 18, 5, 1], [10, -24, 18, 5, 1], [27, 16, 1, 5, 16], [-27, -16, 1, 5, 16]
  ];
  for (const [x, z, w, h, d] of partitions) addWall(x, z, w, h, d, wallMat);

  // hidden room and door gap
  addWall(24, 28, 14, 5, 1, wallMat);
  addWall(24, 18, 14, 5, 1, wallMat);

  makeBed(-28, 28); makeBed(28, -27, Math.PI / 2); makeBed(-8, -20); makeBed(8, 22, Math.PI / 2);
  makeCloset(-30, 6); makeCloset(30, -8); makeCloset(-4, 30); makeCloset(3, -30);
  makeDrawer(-18, -6); makeDrawer(12, 10); makeDrawer(28, 22); makeDrawer(-26, -24);
  makeStairs(-32, -32, 8, Math.PI / 4);

  const gate = new THREE.Mesh(new THREE.BoxGeometry(6, 3.2, 0.7), new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.9 }));
  gate.position.set(0, 1.6, -34.2);
  gate.name = 'exitGate';
  scene.add(gate);
  world.props.push(gate);

  scene.add(new THREE.HemisphereLight(0x4c5f77, 0x0f0d0a, 0.18));
  const moon = new THREE.DirectionalLight(0x91a8d9, 0.35);
  moon.position.set(20, 24, -11);
  moon.castShadow = true;
  scene.add(moon);

  for (let i = 0; i < 14; i++) {
    const lamp = new THREE.PointLight(i % 2 ? 0x7f1d1d : 0x475569, 0.5, 16);
    lamp.position.set((Math.random() - 0.5) * 56, 3, (Math.random() - 0.5) * 56);
    scene.add(lamp);
  }
}

function spawnItems() {
  const candidate = [
    new THREE.Vector3(-26, 0.9, 25), new THREE.Vector3(28, 0.9, -25), new THREE.Vector3(-2, 0.9, -10),
    new THREE.Vector3(7, 0.9, 26), new THREE.Vector3(-26, 0.9, -2), new THREE.Vector3(23, 0.9, 20),
    new THREE.Vector3(-10, 0.9, 10), new THREE.Vector3(17, 0.9, -8)
  ].sort(() => Math.random() - 0.5);

  for (let i = 0; i < game.keysTotal; i++) {
    const key = createKeyMesh();
    key.position.copy(candidate[i]);
    key.userData.collected = false;
    scene.add(key);
    world.keys.push(key);
  }

  for (let i = 0; i < 4; i++) {
    const ammo = createGunPickup();
    ammo.position.copy(candidate[i + 3]);
    ammo.userData.collected = false;
    scene.add(ammo);
    world.ammoPickups.push(ammo);
  }
}

function buildScaryGhost() {
  const g = new THREE.Group();
  const skin = new THREE.MeshStandardMaterial({ color: 0xd9d3d0, roughness: 0.92 });
  const blood = new THREE.MeshStandardMaterial({ color: 0x5b0715, roughness: 0.55, metalness: 0.2 });

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.45, 1.4, 6, 10), skin);
  torso.position.y = 1.4;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.35, 18, 18), skin);
  head.position.y = 2.35;

  const armL = new THREE.Mesh(new THREE.CapsuleGeometry(0.12, 0.85), skin);
  const armR = armL.clone();
  armL.position.set(-0.52, 1.4, 0); armR.position.set(0.52, 1.4, 0);

  const legL = new THREE.Mesh(new THREE.CapsuleGeometry(0.14, 0.92), skin);
  const legR = legL.clone();
  legL.position.set(-0.2, 0.58, 0); legR.position.set(0.2, 0.58, 0);

  const tooth1 = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.18, 6), new THREE.MeshStandardMaterial({ color: 0xf8fafc }));
  const tooth2 = tooth1.clone();
  tooth1.position.set(-0.06, 2.07, 0.27); tooth2.position.set(0.06, 2.07, 0.27);

  const bloodStripe = new THREE.Mesh(new THREE.BoxGeometry(0.22, 1.15, 0.07), blood);
  bloodStripe.position.set(0.1, 1.65, 0.34);

  g.add(torso, head, armL, armR, legL, legR, tooth1, tooth2, bloodStripe);
  g.position.copy(ghost.pos);
  scene.add(g);
  ghost.root = g;
}

function worldCollision(nextPos) {
  const pBox = new THREE.Box3(new THREE.Vector3(nextPos.x - 0.33, 0.1, nextPos.z - 0.33), new THREE.Vector3(nextPos.x + 0.33, 1.75, nextPos.z + 0.33));
  for (const w of world.walls) if (w.intersectsBox(pBox)) return true;
  return false;
}

function setupControls() {
  const joystick = document.getElementById('joystick');
  const stick = document.getElementById('stick');
  const lookZone = document.getElementById('lookZone');

  let joyId = null;
  let joyOrigin = null;

  joystick.addEventListener('touchstart', (e) => {
    const t = e.changedTouches[0];
    joyId = t.identifier;
    const r = joystick.getBoundingClientRect();
    joyOrigin = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }, { passive: true });

  joystick.addEventListener('touchmove', (e) => {
    for (const t of e.changedTouches) {
      if (t.identifier !== joyId) continue;
      const dx = t.clientX - joyOrigin.x;
      const dy = t.clientY - joyOrigin.y;
      const len = Math.min(48, Math.hypot(dx, dy));
      const ang = Math.atan2(dy, dx);
      const sx = Math.cos(ang) * len;
      const sy = Math.sin(ang) * len;
      stick.style.transform = `translate(calc(-50% + ${sx}px), calc(-50% + ${sy}px))`;
      player.move.set(sx / 48, sy / 48);
    }
  }, { passive: false });

  const releaseJoy = () => {
    joyId = null;
    player.move.set(0, 0);
    stick.style.transform = 'translate(-50%, -50%)';
  };
  joystick.addEventListener('touchend', releaseJoy);
  joystick.addEventListener('touchcancel', releaseJoy);

  let lookId = null;
  let lastLook = null;
  const doLook = (dx, dy) => {
    player.yaw -= dx * 0.003;
    player.pitch -= dy * 0.0022;
    player.pitch = THREE.MathUtils.clamp(player.pitch, -1.2, 1.2);
  };

  lookZone.addEventListener('touchstart', (e) => {
    const t = e.changedTouches[0];
    lookId = t.identifier;
    lastLook = { x: t.clientX, y: t.clientY };
  }, { passive: true });

  lookZone.addEventListener('touchmove', (e) => {
    for (const t of e.changedTouches) {
      if (t.identifier !== lookId || !lastLook) continue;
      e.preventDefault();
      const dx = t.clientX - lastLook.x;
      const dy = t.clientY - lastLook.y;
      doLook(dx, dy);
      lastLook = { x: t.clientX, y: t.clientY };
    }
  }, { passive: false });
  lookZone.addEventListener('touchend', () => { lookId = null; lastLook = null; });

  // desktop free-look
  renderer.domElement.addEventListener('click', () => {
    if (!game.started || game.over || game.won) return;
    renderer.domElement.requestPointerLock?.();
  });
  document.addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === renderer.domElement) doLook(e.movementX, e.movementY);
  });

  const runBtn = document.getElementById('runBtn');
  runBtn.addEventListener('touchstart', () => game.running = true);
  runBtn.addEventListener('touchend', () => game.running = false);

  const crouchBtn = document.getElementById('crouchBtn');
  crouchBtn.addEventListener('touchstart', () => {
    game.crouching = !game.crouching;
    crouchBtn.style.background = game.crouching ? 'rgba(31,160,94,.5)' : 'rgba(255,255,255,.14)';
    camera.position.y = game.crouching ? 1.0 : 1.7;
  });

  const actionBtn = document.getElementById('actionBtn');
  actionBtn.addEventListener('touchstart', () => {
    tone(500, 0.06, 'triangle', 0.03);
  });

  const holdBtn = document.getElementById('holdBreathBtn');
  holdBtn.addEventListener('touchstart', () => {
    game.holdBreath = true;
    holdBtn.style.background = 'rgba(31,160,94,.45)';
  });
  holdBtn.addEventListener('touchend', () => {
    game.holdBreath = false;
    holdBtn.style.background = 'rgba(255,255,255,.14)';
  });

  const invBtn = document.getElementById('inventoryBtn');
  const closeInvBtn = document.getElementById('closeInventoryBtn');
  const toggleInventory = () => {
    game.inventoryOpen = !game.inventoryOpen;
    ui.inventoryOverlay.classList.toggle('visible', game.inventoryOpen);
  };
  invBtn.addEventListener('touchstart', toggleInventory);
  closeInvBtn.addEventListener('click', toggleInventory);

  const fireBtn = document.getElementById('fireBtn');
  const fire = () => {
    if (!game.started || game.over || game.won || game.ammo <= 0) return;
    game.ammo -= 1;
    tone(760, 0.13, 'square', 0.06, -200);
    noiseBurst(0.08, 0.015);

    const fwd = new THREE.Vector3(Math.sin(player.yaw), 0, Math.cos(player.yaw)).normalize();
    const toGhost = ghost.pos.clone().sub(player.pos);
    const dist = toGhost.length();
    if (dist < 16 && fwd.dot(toGhost.normalize()) > 0.93) {
      ghost.stunned = 3.8;
      ghost.rage = false;
      tone(140, 0.35, 'sawtooth', 0.06, -40);
    }
  };
  fireBtn.addEventListener('touchstart', fire);
  window.addEventListener('keydown', (e) => { if (e.code === 'Space') fire(); });
}

function inClosetCover() {
  for (const c of world.closets) {
    if (c.position.distanceTo(player.pos) < 2.2) return true;
  }
  return false;
}


function updateDayNightAndSanity(dt) {
  game.daySeconds += dt;
  const cycle = game.daySeconds % (20 * 60);
  const isNight = cycle > 12 * 60;
  const isTwilight = cycle > 8 * 60 && cycle <= 12 * 60;

  ghost.speed = isNight ? 1.7 : isTwilight ? 1.5 : 1.35;
  ghost.rageSpeed = isNight ? 2.7 : isTwilight ? 2.45 : 2.25;

  const dist = ghost.pos.distanceTo(player.pos);
  let sanityDelta = 0;
  if (isNight) sanityDelta -= dt * 1.2;
  if (dist < 8) sanityDelta -= dt * (8 - dist) * 0.8;
  if (game.holdBreath) sanityDelta -= dt * 2.1;
  if (!ghost.rage && dist > 13) sanityDelta += dt * 0.55;
  game.sanity = THREE.MathUtils.clamp(game.sanity + sanityDelta, 0, 100);

  if (game.sanity < 30) {
    camera.rotation.z = Math.sin(performance.now() * 0.003) * 0.01;
  } else {
    camera.rotation.z = 0;
  }

  const currentDay = Math.floor(game.daySeconds / (20 * 60)) + 1;
  game.dayIndex = Math.min(7, currentDay);
}

function ambientAndThreatSounds(dt) {
  if (!game.started || game.over || game.won) return;
  ghost.howlTimer -= dt;
  const d = ghost.pos.distanceTo(player.pos);
  if (ghost.howlTimer <= 0) {
    tone(45 + Math.random() * 10, 1.7, 'sawtooth', 0.02, Math.random() * 15);
    if (d < 12) {
      tone(260 + Math.random() * 90, 0.25, 'triangle', 0.028, -30);
      noiseBurst(0.2, 0.02);
    }
    if (ghost.rage && d < 9) tone(90, 0.2, 'square', 0.035, -30);
    ghost.howlTimer = 0.85 + Math.random() * 0.6;
  }
}

function updateItems() {
  for (const k of world.keys) {
    if (k.userData.collected) continue;
    k.rotation.y += 0.04;
    if (k.position.distanceTo(player.pos) < 1.4) {
      k.userData.collected = true;
      k.visible = false;
      game.keysFound += 1;
      tone(980, 0.15, 'triangle', 0.06, 120);
      if (game.keysFound >= game.keysTotal) {
        game.gateUnlocked = true;
        setObjective('Gate unlocked. Reach basement exit.');
        const gate = world.props.find((p) => p.name === 'exitGate');
        if (gate) gate.material.color.set(0x1fa05e);
      }
    }
  }

  for (const a of world.ammoPickups) {
    if (a.userData.collected) continue;
    a.rotation.y += 0.03;
    if (a.position.distanceTo(player.pos) < 1.5) {
      a.userData.collected = true;
      a.visible = false;
      game.ammo += 3;
      tone(620, 0.12, 'square', 0.05, 60);
    }
  }
}

function updatePlayer(dt) {
  const forward = new THREE.Vector3(Math.sin(player.yaw), 0, Math.cos(player.yaw));
  const right = new THREE.Vector3(forward.z, 0, -forward.x);
  const moveDir = new THREE.Vector3().addScaledVector(forward, -player.move.y).addScaledVector(right, player.move.x);
  if (moveDir.lengthSq() > 0.001) moveDir.normalize();

  let speed = game.crouching ? 1.2 : 2.1;
  if (game.running && game.stamina > 0 && !game.crouching && moveDir.lengthSq() > 0.01) {
    speed = 4.0;
    game.stamina = Math.max(0, game.stamina - dt * 24);
  } else {
    game.stamina = Math.min(100, game.stamina + dt * 13);
  }

  const moving = moveDir.lengthSq() > 0.002;
  game.noiseLevel = moving ? (game.running ? 1 : game.crouching ? 0.2 : 0.5) : Math.max(0, game.noiseLevel - dt * 1.8);

  const candidate = player.pos.clone().addScaledVector(moveDir, speed * dt);
  candidate.x = THREE.MathUtils.clamp(candidate.x, world.boundsMin.x + 1.2, world.boundsMax.x - 1.2);
  candidate.z = THREE.MathUtils.clamp(candidate.z, world.boundsMin.z + 1.2, world.boundsMax.z - 1.2);
  if (!worldCollision(candidate)) player.pos.copy(candidate);

  camera.position.copy(player.pos);
  camera.rotation.order = 'YXZ';
  camera.rotation.y = player.yaw;
  camera.rotation.x = player.pitch;
}

function hasLineOfSight(from, to) {
  const ray = new THREE.Raycaster(from, to.clone().sub(from).normalize(), 0, from.distanceTo(to));
  const hits = ray.intersectObjects(world.props);
  return hits.length === 0;
}

function updateGhost(dt) {
  if (!ghost.root || game.over || game.won) return;

  if (ghost.stunned > 0) {
    ghost.stunned -= dt;
    ghost.root.rotation.z = Math.sin(performance.now() * 0.02) * 0.12;
    return;
  }

  const toPlayer = player.pos.clone().sub(ghost.pos);
  const dist = toPlayer.length();
  const hears = dist < (game.noiseLevel > 0.66 ? 15 : game.noiseLevel > 0.33 ? 8 : 5);
  const visible = dist < 17 && hasLineOfSight(ghost.pos, player.pos) && !(game.crouching && inClosetCover());

  if (visible || hears) {
    ghost.rage = true;
    ghost.cooldown = 5;
  } else {
    ghost.cooldown -= dt;
    if (ghost.cooldown <= 0) ghost.rage = false;
  }

  const stepSpeed = ghost.rage ? ghost.rageSpeed : ghost.speed;
  if (dist > 1.35) {
    ghost.pos.addScaledVector(toPlayer.normalize(), stepSpeed * dt);
  }

  ghost.pos.x = THREE.MathUtils.clamp(ghost.pos.x, world.boundsMin.x + 1.2, world.boundsMax.x - 1.2);
  ghost.pos.z = THREE.MathUtils.clamp(ghost.pos.z, world.boundsMin.z + 1.2, world.boundsMax.z - 1.2);

  ghost.footPhase += dt * (ghost.rage ? 9 : 6);
  const swing = Math.sin(ghost.footPhase) * 0.24;
  ghost.root.children[2].rotation.x = swing;
  ghost.root.children[3].rotation.x = -swing;
  ghost.root.children[4].rotation.x = -swing;
  ghost.root.children[5].rotation.x = swing;

  ghost.root.position.copy(ghost.pos);
  ghost.root.lookAt(player.pos.x, ghost.pos.y + 1, player.pos.z);

  if (dist < 1.45) {
    game.health -= 1;
    if (game.health > 0 && (game.daySeconds % (20 * 60)) < (12 * 60)) {
      player.pos.set((Math.random() - 0.5) * 20, player.pos.y, (Math.random() - 0.5) * 20);
      ghost.pos.set((Math.random() - 0.5) * 20, ghost.pos.y, -24);
      tone(110, 0.2, 'triangle', 0.06);
      return;
    }
    game.over = true;
    ui.deathOverlay.classList.add('visible');
    let t = 0;
    const killAnim = () => {
      if (!game.over) return;
      t += 0.05;
      camera.fov = THREE.MathUtils.lerp(72, 125, t);
      camera.updateProjectionMatrix();
      if (t < 1) requestAnimationFrame(killAnim);
    };
    killAnim();
    tone(85, 0.45, 'sawtooth', 0.08, -20);
    noiseBurst(0.35, 0.03);
  }
}

function checkWin() {
  if (!game.gateUnlocked) return;
  if (player.pos.distanceTo(new THREE.Vector3(0, 1.7, -33)) < 2.2) {
    game.won = true;
    ui.winOverlay.classList.add('visible');
    tone(1180, 0.2, 'triangle', 0.08, 80);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.033);

  if (game.started && !game.over && !game.won) {
    updatePlayer(dt);
    updateGhost(dt);
    updateItems();
    updateDayNightAndSanity(dt);
    ambientAndThreatSounds(dt);
    checkWin();
    updateHUD();
  }

  renderer.render(scene, camera);
}

function simulateLoadAndStartMenu() {
  const hints = ['Waking the house...', 'Lighting ritual candles...', 'Sharpening the hunter teeth...', 'Loading blood corridors...'];
  let p = 0;
  let i = 0;
  const timer = setInterval(() => {
    p += 8 + Math.random() * 9;
    ui.loadingFill.style.width = `${Math.min(100, p)}%`;
    ui.loadingHint.textContent = hints[i % hints.length];
    i++;
    tone(140 + Math.random() * 40, 0.08, 'triangle', 0.01);
    if (p >= 100) {
      clearInterval(timer);
      game.loadingDone = true;
      setTimeout(() => {
        ui.loadingOverlay.classList.remove('visible');
        ui.startOverlay.classList.add('visible');
        noiseBurst(0.25, 0.015);
      }, 500);
    }
  }, 260);
}

function init() {
  setupWorld();
  spawnItems();
  buildScaryGhost();
  setupControls();
  setObjective('Find 3 relic keys');
  updateHUD();
  animate();
  simulateLoadAndStartMenu();

  document.getElementById('startBtn').addEventListener('click', async () => {
    if (!game.loadingDone) return;
    if (audioCtx.state !== 'running') await audioCtx.resume();
    ui.startOverlay.classList.remove('visible');
    game.started = true;
    tone(250, 0.25, 'sawtooth', 0.03, -20);
  });

  const restart = () => window.location.reload();
  document.getElementById('retryBtn').addEventListener('click', restart);
  document.getElementById('playAgainBtn').addEventListener('click', restart);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

init();
