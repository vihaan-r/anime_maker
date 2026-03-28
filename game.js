import * as THREE from 'https://unpkg.com/three@0.166.1/build/three.module.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x07080b, 0.06);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 400);
camera.position.set(0, 1.7, 0);

const listener = new THREE.AudioListener();
camera.add(listener);

const ui = {
  objective: document.getElementById('objective'),
  keys: document.getElementById('keys'),
  stamina: document.getElementById('stamina'),
  threat: document.getElementById('threat'),
  start: document.getElementById('startOverlay'),
  death: document.getElementById('deathOverlay'),
  win: document.getElementById('winOverlay')
};

const game = {
  started: false,
  over: false,
  won: false,
  totalKeys: 3,
  keysFound: 0,
  stamina: 100,
  running: false,
  hiding: false,
  velocityY: 0,
  gravity: -16,
  speed: 2,
  runSpeed: 3.8,
  ghostAwareness: 0,
  gateUnlocked: false
};

const world = {
  walls: [],
  props: [],
  keys: [],
  houseMin: new THREE.Vector3(-24, 0, -24),
  houseMax: new THREE.Vector3(24, 0, 24)
};

const player = {
  pos: new THREE.Vector3(0, 1.7, 17),
  yaw: Math.PI,
  pitch: 0,
  lookTouchId: null,
  moveTouchId: null,
  moveVec: new THREE.Vector2(),
  lookPrev: null
};

const ghost = {
  mesh: null,
  pos: new THREE.Vector3(0, 0.8, -16),
  speed: 1.4,
  rageSpeed: 2.2,
  rage: false,
  cooldown: 0
};

const clock = new THREE.Clock();
const loader = new THREE.TextureLoader();

function makeTextures() {
  const floorTex = loader.load('https://threejs.org/examples/textures/hardwood2_diffuse.jpg');
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping;
  floorTex.repeat.set(12, 12);

  const wallTex = loader.load('https://threejs.org/examples/textures/brick_diffuse.jpg');
  wallTex.wrapS = wallTex.wrapT = THREE.RepeatWrapping;
  wallTex.repeat.set(10, 2);

  const roofTex = loader.load('https://threejs.org/examples/textures/uv_grid_opengl.jpg');
  roofTex.wrapS = roofTex.wrapT = THREE.RepeatWrapping;
  roofTex.repeat.set(8, 8);

  return { floorTex, wallTex, roofTex };
}

const tex = makeTextures();

function setupLighting() {
  scene.add(new THREE.HemisphereLight(0x5f7592, 0x0d0a07, 0.2));

  const moon = new THREE.DirectionalLight(0x9cb1d0, 0.35);
  moon.position.set(16, 20, -8);
  moon.castShadow = true;
  moon.shadow.mapSize.set(1024, 1024);
  scene.add(moon);

  const redPulse = new THREE.PointLight(0x8b1d2b, 0.6, 24);
  redPulse.position.set(-17, 2, -16);
  scene.add(redPulse);

  setInterval(() => {
    redPulse.intensity = 0.15 + Math.random() * 0.8;
  }, 180);
}

function makeWall(x, z, w, h, d) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ map: tex.wallTex, roughness: 0.9 })
  );
  mesh.position.set(x, h / 2, z);
  mesh.castShadow = mesh.receiveShadow = true;
  scene.add(mesh);
  world.walls.push(new THREE.Box3().setFromObject(mesh));
}

function generateMansion() {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(54, 54),
    new THREE.MeshStandardMaterial({ map: tex.floorTex, roughness: 0.88 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const ceiling = new THREE.Mesh(
    new THREE.PlaneGeometry(54, 54),
    new THREE.MeshStandardMaterial({ map: tex.roofTex, color: 0x333944, roughness: 1 })
  );
  ceiling.position.y = 4;
  ceiling.rotation.x = Math.PI / 2;
  scene.add(ceiling);

  makeWall(0, -27, 54, 4, 1);
  makeWall(0, 27, 54, 4, 1);
  makeWall(-27, 0, 1, 4, 54);
  makeWall(27, 0, 1, 4, 54);

  const lines = [
    [-14, 0, 1, 4, 38], [14, 0, 1, 4, 38], [0, -9, 28, 4, 1], [0, 9, 28, 4, 1],
    [-7, 18, 14, 4, 1], [7, -18, 14, 4, 1], [-20, -9, 1, 4, 16], [20, 9, 1, 4, 16]
  ];
  for (const [x, z, w, h, d] of lines) makeWall(x, z, w, h, d);

  const gate = new THREE.Mesh(
    new THREE.BoxGeometry(6, 3, 0.6),
    new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.4 })
  );
  gate.position.set(0, 1.5, -26.2);
  gate.name = 'exitGate';
  gate.castShadow = true;
  scene.add(gate);
  world.props.push(gate);

  for (let i = 0; i < 24; i++) {
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(2, 1.2 + Math.random() * 1.8, 2),
      new THREE.MeshStandardMaterial({ color: 0x201f29, roughness: 0.95 })
    );
    box.position.set((Math.random() - 0.5) * 44, box.geometry.parameters.height / 2, (Math.random() - 0.5) * 44);
    box.castShadow = box.receiveShadow = true;
    scene.add(box);
    world.walls.push(new THREE.Box3().setFromObject(box));
    world.props.push(box);
  }
}

function spawnKeys() {
  const positions = [
    new THREE.Vector3(-20, 0.55, 19),
    new THREE.Vector3(19, 0.55, 15),
    new THREE.Vector3(-15, 0.55, -19)
  ];

  for (let i = 0; i < game.totalKeys; i++) {
    const relic = new THREE.Mesh(
      new THREE.TorusKnotGeometry(0.33, 0.08, 90, 14),
      new THREE.MeshStandardMaterial({ color: 0x60a5fa, emissive: 0x1d4ed8, emissiveIntensity: 0.8 })
    );
    relic.position.copy(positions[i]);
    relic.userData.collected = false;
    scene.add(relic);
    world.keys.push(relic);
  }
}

function spawnGhost() {
  const mat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.9, emissive: 0x111827 });
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.45, 1.7, 6, 12), mat);
  body.position.copy(ghost.pos);
  body.castShadow = true;
  scene.add(body);
  ghost.mesh = body;
}

function synthSound(freq, length, type = 'sine', gain = 0.08) {
  const audioCtx = listener.context;
  const osc = audioCtx.createOscillator();
  const vol = audioCtx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  vol.gain.value = gain;
  osc.connect(vol);
  vol.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + length);
}

function ambienceLoop() {
  setInterval(() => {
    if (!game.started || game.over || game.won) return;
    synthSound(45 + Math.random() * 20, 1.8, 'sawtooth', 0.025);
    if (Math.random() > 0.6) synthSound(180 + Math.random() * 80, 0.15, 'triangle', 0.03);
  }, 1400);
}

function padInput(element, onMove, onEnd) {
  let id = null;
  let origin = null;
  element.addEventListener('touchstart', (e) => {
    const t = e.changedTouches[0];
    id = t.identifier;
    origin = { x: t.clientX, y: t.clientY };
  }, { passive: false });

  element.addEventListener('touchmove', (e) => {
    for (const t of e.changedTouches) {
      if (t.identifier !== id) continue;
      e.preventDefault();
      const dx = t.clientX - origin.x;
      const dy = t.clientY - origin.y;
      onMove(dx, dy);
    }
  }, { passive: false });

  const endFn = () => {
    id = null;
    onEnd();
  };
  element.addEventListener('touchend', endFn);
  element.addEventListener('touchcancel', endFn);
}

function setupControls() {
  padInput(document.getElementById('movePad'), (dx, dy) => {
    player.moveVec.set(
      Math.max(-1, Math.min(1, dx / 46)),
      Math.max(-1, Math.min(1, dy / 46))
    );
  }, () => player.moveVec.set(0, 0));

  padInput(document.getElementById('lookPad'), (dx, dy) => {
    player.yaw -= dx * 0.003;
    player.pitch -= dy * 0.002;
    player.pitch = Math.max(-1.1, Math.min(1.1, player.pitch));
  }, () => {});

  const runBtn = document.getElementById('runBtn');
  runBtn.addEventListener('touchstart', () => game.running = true);
  runBtn.addEventListener('touchend', () => game.running = false);

  const hideBtn = document.getElementById('hideBtn');
  hideBtn.addEventListener('touchstart', () => {
    game.hiding = !game.hiding;
    hideBtn.style.background = game.hiding ? 'rgba(34,197,94,.45)' : 'rgba(255,255,255,.16)';
    camera.position.y = game.hiding ? 1.0 : 1.7;
  });
}

function setObjective(text) {
  ui.objective.textContent = text;
}

function updateHUD() {
  ui.keys.textContent = `${game.keysFound} / ${game.totalKeys}`;
  ui.stamina.textContent = `${Math.round(game.stamina)}%`;
  ui.threat.textContent = ghost.rage ? 'Hunting' : 'Searching';
}

function tryCollectKeys() {
  for (const key of world.keys) {
    if (key.userData.collected) continue;
    key.rotation.y += 0.03;
    const d = key.position.distanceTo(player.pos);
    if (d < 1.45) {
      key.userData.collected = true;
      key.visible = false;
      game.keysFound += 1;
      synthSound(780, 0.12, 'square', 0.08);
      if (game.keysFound === game.totalKeys) {
        game.gateUnlocked = true;
        setObjective('Gate unlocked! Reach the basement exit.');
        const gate = world.props.find((p) => p.name === 'exitGate');
        if (gate) gate.material.color.set(0x16a34a);
      }
      updateHUD();
    }
  }
}

function lineBlocked(from, to) {
  const ray = new THREE.Raycaster(from, to.clone().sub(from).normalize(), 0, from.distanceTo(to));
  const hits = ray.intersectObjects(world.props);
  return hits.length > 0;
}

function updateGhost(dt) {
  if (!ghost.mesh || game.over || game.won) return;

  const toPlayer = player.pos.clone().sub(ghost.pos);
  const dist = toPlayer.length();
  const canSee = dist < 14 && !lineBlocked(ghost.pos, player.pos) && !game.hiding;

  if (canSee) {
    ghost.rage = true;
    game.ghostAwareness = Math.min(100, game.ghostAwareness + dt * 35);
    ghost.cooldown = 4;
  } else {
    ghost.cooldown -= dt;
    if (ghost.cooldown <= 0) ghost.rage = false;
    game.ghostAwareness = Math.max(0, game.ghostAwareness - dt * 12);
  }

  const speed = ghost.rage ? ghost.rageSpeed : ghost.speed;
  if (dist > 1.2) {
    toPlayer.normalize();
    ghost.pos.addScaledVector(toPlayer, speed * dt);
  }

  ghost.pos.x = THREE.MathUtils.clamp(ghost.pos.x, world.houseMin.x + 1, world.houseMax.x - 1);
  ghost.pos.z = THREE.MathUtils.clamp(ghost.pos.z, world.houseMin.z + 1, world.houseMax.z - 1);

  ghost.mesh.position.copy(ghost.pos);
  ghost.mesh.position.y = 0.8 + Math.sin(performance.now() * 0.005) * 0.15;
  ghost.mesh.lookAt(player.pos.x, ghost.mesh.position.y, player.pos.z);

  if (dist < 1.5) triggerDeath();
}

function movePlayer(dt) {
  const dir = new THREE.Vector3();
  const forward = new THREE.Vector3(Math.sin(player.yaw), 0, Math.cos(player.yaw));
  const right = new THREE.Vector3(forward.z, 0, -forward.x);

  dir.addScaledVector(forward, -player.moveVec.y);
  dir.addScaledVector(right, player.moveVec.x);
  if (dir.lengthSq() > 0) dir.normalize();

  let speed = game.speed;
  if (game.running && game.stamina > 0 && dir.lengthSq() > 0.01 && !game.hiding) {
    speed = game.runSpeed;
    game.stamina = Math.max(0, game.stamina - 24 * dt);
  } else {
    game.stamina = Math.min(100, game.stamina + 12 * dt);
  }

  const old = player.pos.clone();
  player.pos.addScaledVector(dir, speed * dt);
  player.pos.x = THREE.MathUtils.clamp(player.pos.x, world.houseMin.x + 1.2, world.houseMax.x - 1.2);
  player.pos.z = THREE.MathUtils.clamp(player.pos.z, world.houseMin.z + 1.2, world.houseMax.z - 1.2);

  const playerBox = new THREE.Box3(
    new THREE.Vector3(player.pos.x - 0.35, 0, player.pos.z - 0.35),
    new THREE.Vector3(player.pos.x + 0.35, 1.8, player.pos.z + 0.35)
  );
  for (const wall of world.walls) {
    if (wall.intersectsBox(playerBox)) {
      player.pos.copy(old);
      break;
    }
  }

  camera.position.copy(player.pos);
  camera.rotation.order = 'YXZ';
  camera.rotation.y = player.yaw;
  camera.rotation.x = player.pitch;
}

function checkWinCondition() {
  if (!game.gateUnlocked) return;
  if (player.pos.distanceTo(new THREE.Vector3(0, 1.7, -24.6)) < 2.4) {
    game.won = true;
    ui.win.classList.add('visible');
    synthSound(1200, 0.22, 'sine', 0.09);
  }
}

function triggerDeath() {
  if (game.over || game.won) return;
  game.over = true;
  ui.death.classList.add('visible');

  // quick kill animation
  const start = performance.now();
  const killZoom = () => {
    const t = (performance.now() - start) / 700;
    camera.fov = THREE.MathUtils.lerp(70, 125, Math.min(1, t));
    camera.updateProjectionMatrix();
    if (t < 1) requestAnimationFrame(killZoom);
  };
  killZoom();
  synthSound(90, 0.4, 'sawtooth', 0.11);
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(0.033, clock.getDelta());

  if (game.started && !game.over && !game.won) {
    movePlayer(dt);
    updateGhost(dt);
    tryCollectKeys();
    checkWinCondition();
    updateHUD();
  }

  renderer.render(scene, camera);
}

function hardReset() {
  window.location.reload();
}

function init() {
  setupLighting();
  generateMansion();
  spawnKeys();
  spawnGhost();
  setupControls();
  updateHUD();
  setObjective('Find 3 relic keys');
  ambienceLoop();

  document.getElementById('startBtn').addEventListener('click', () => {
    ui.start.classList.remove('visible');
    game.started = true;
    synthSound(280, 0.2, 'triangle', 0.04);
  });
  document.getElementById('retryBtn').addEventListener('click', hardReset);
  document.getElementById('playAgainBtn').addEventListener('click', hardReset);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  animate();
}

init();
