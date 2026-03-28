# Whispering Shadows (Mobile Horror Escape Prototype)

This repository contains a **3D mobile horror escape game prototype** inspired by tense house-survival gameplay loops.

For the full production-grade design and build roadmap, see `WHISPERING_SHADOWS_AAA_PLAN.md` and `design/whispering_shadows_game_config.json`.

## Latest gameplay upgrade

- New horror branding/logo: **Whispering Shadows**.
- New loading/start sequence with animated progress + creepy audio pulses.
- Real joystick movement + dedicated crouch and fire buttons.
- Free-look controls (touch drag on right zone + pointer-lock mouse look).
- Scarier humanoid ghost with blood/tooth details and walking animation.
- Better proximity-based ghost ambience/sound pressure.
- Expanded map props: closets, beds, drawers, stairs, hidden-room layout.
- Gun/ammo pickups and ghost stun mechanic.
- Randomized key spawn points with key-shaped 3D pickups.
- Procedural textures generated at runtime from scratch (no external texture downloads).

## What is already built

- Full-screen 3D environment with textured floors/walls/ceiling.
- Large mansion map with corridors, rooms, obstacles, and an exit gate.
- Touch-first controls for mobile (move joystick, look pad, run/hide buttons).
- Horror loop: collect 3 relic keys, unlock gate, avoid a roaming ghost AI, escape.
- UI overlays (intro/start sequence, HUD objectives, death and victory screens).
- Kill animation and dynamic ambient/feedback sound synthesis.

## Run locally

```bash
python3 -m http.server 4173
```

Open: `http://localhost:4173`

---

## How to get your Android APK (exact steps)


## I just want the APK file (simple)

I understand — here is the shortest path:

1. Push this repo to GitHub.
2. Open **Actions** tab.
3. Run workflow: **Build Android APK**.
4. Wait for it to finish.
5. Download artifact: **phantom-lockdown-debug-apk**.
6. The file inside is your APK: `app-debug.apk`.

> Note: I cannot directly attach binary files in this chat, so the workflow above is the easiest "one-click" APK download path.

---

### Fastest method (scripted)

From the repo root:

```bash
./scripts/build_android_apk.sh
```

If your Android SDK is configured correctly, your debug APK will be created at:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

### Manual method

1. Install:
   - Node.js (20+ recommended)
   - Android Studio (with Android SDK + platform tools)
2. Install Capacitor packages:

```bash
npm init -y
npm i -D @capacitor/core @capacitor/cli @capacitor/android
```

3. Initialize Capacitor (one-time):

```bash
npx cap init "Phantom Lockdown" "com.phantom.lockdown" --web-dir www
```

4. Add Android platform (one-time):

```bash
npx cap add android
```

Before Capacitor sync/build, copy web files into `www`:

```bash
rm -rf www
mkdir -p www
cp index.html game.js styles.css www/
```

5. Sync web assets into Android project:

```bash
npx cap sync android
```

6. Build APK using Gradle:

```bash
cd android
./gradlew assembleDebug
```

7. APK location:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

---


### If you do not see the action

Do these checks:

1. Make sure this file exists in your repo: `.github/workflows/build-apk.yml`.
2. Push to `main` (or open a PR) at least once — the workflow now auto-appears on push/PR.
3. In GitHub repo settings, ensure **Actions** are enabled.
4. If this is a fork, enable Actions in the fork (forks can have Actions disabled by default).

## If APK build fails

Common fixes:

- If you get Kotlin duplicate class errors (for `kotlin-stdlib-jdk7/jdk8`), remove `kotlin-stdlib-jdk7/jdk8` and force one Kotlin stdlib version (the build script now patches this automatically).
- If Gradle says `invalid source release: 21`, use JDK 21 (not 17). In GitHub Actions, set `java-version: '21'`.
- If you get `"." is not a valid value for webDir`, set `webDir` to `www` (not `.`) and copy files into `www` before `npx cap sync android`.
- Ensure `ANDROID_HOME` is set and SDK tools are installed.
- In Android Studio, install a recent Android SDK Platform + Build-Tools.
- Run once in Android Studio to auto-accept SDK components.
- If Gradle fails from CLI, use:

```bash
npx cap open android
```

Then build from Android Studio: `Build > Build APK(s)`.

## Gameplay notes

- Difficulty target: hard but fair.
- Ghost enters hunt mode when line-of-sight is established and you are not hiding.
- Running drains stamina; hiding lowers ghost detection chance.
- Win by unlocking and reaching the basement gate.
