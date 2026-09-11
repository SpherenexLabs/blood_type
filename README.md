# Blood Group Detector (React)

A webcam blood-group card analyser that runs entirely in the browser. This is a
port of the old Flask app (`blood_type/blood.py` +
`blood_type/templates/blood_dashboard.html`) — there is no Python server any
more, and everything lives in [src/App.jsx](src/App.jsx).

## Run it

```bash
npm install
npm run dev
```

Then open the printed URL. Camera access needs a secure context, so use
`http://localhost:5173` (localhost counts as secure) or serve the build over
HTTPS — a plain LAN IP such as `http://192.168.1.20:5173` will be refused by the
browser.

## How the dashboard works

- **Auto camera.** On load it asks for camera permission, enumerates every video
  input, and opens them in preference order — a USB/external camera first, a
  built-in lid camera next, virtual cameras (OBS, DroidCam…) last. This mirrors
  the old `CAMERA_INDEX=1` default. Each candidate is opened and checked for real
  frames before it is accepted, and the list is rescanned automatically when a
  camera is plugged in or removed.
- **Picking a camera yourself.** The dropdown lists every camera the device
  reports, labelled with its kind (front / back / USB / built-in) and how many
  were found; picking one switches the feed immediately and **saves that
  choice** — reloads, restarts and hot reloads keep using it, and the line under
  "Camera" reads `your choice, saved` instead of `auto-selected`. Nothing
  overrides a working choice: the auto-scan only runs when there is no saved
  camera, and if the saved one is unplugged it says so before falling back (and
  returns to it when you plug it back in). **Switch** hops to the next camera in
  one tap (handy for front/back on a phone), **Auto** forgets the saved choice
  and goes back to automatic, **Rescan** refreshes the list. Resolution and frame
  rate are requested as preferences only, so a camera that cannot manage them is
  still opened rather than skipped.
- **Frame rate.** The preview is the raw `<video>` element requested at
  1280x720 / 60fps, so frames go camera → compositor without passing through
  JavaScript (the old MJPEG `/video_feed` re-encoded every frame). The badge
  reports the frames the camera actually presented, via
  `requestVideoFrameCallback`.
- **Capture & detect.** Grabs the current frame, sends it to the Roboflow model,
  scores each well, and shows the group plus the annotated frame, the three
  cropped wells and their clump masks.
- **Live detect.** Keeps the analysis running: Roboflow re-locates the wells
  about every 1.8s, while the agglutination pass cycles through A, B and D one
  well per tick (~50ms each) so the preview never stalls. Boxes are redrawn on an
  overlay canvas every presented frame.

## Detection pipeline

Unchanged from the Python version, reimplemented on canvas pixels:

1. Roboflow `blood-group-detection-4yvdx/1` locates the Anti-A / Anti-B / Anti-D
   regions (predictions below 0.35 confidence are dropped).
2. If the model misses a region, the dark-card fallback finds the wide dark test
   card — by contour, or by row/column projection profiles when the card merges
   into an adjacent dark panel — and splits it into three wells.
3. Each well is resized to 300x300, blurred, pushed through CIE L\*a\*b\* so CLAHE
   can lift only the lightness channel, thresholded adaptively, opened and
   closed, and scored:
   `0.60 * clumpRatio + 0.25 * texture + 0.15 * clumpCount` (normalised).
   At or above 0.12 the well reads POSITIVE.
4. A/B/D positives map to the group: `A+`, `O-`, `AB+`, and so on.

The port was checked against OpenCV 4.11 on identical pixel buffers: the worst
score difference is 0.0004, and the card-layout fallback returns byte-identical
boxes.

## Configuration

The Roboflow key is compiled into the bundle, which is inherent to calling the
API from the browser — anyone who loads the page can read it. Use a restricted
key, and rotate it if the page goes public. Override any default with a
`.env.local`:

```
VITE_ROBOFLOW_API_KEY=...
VITE_ROBOFLOW_MODEL_ID=blood-group-detection-4yvdx/1
VITE_ROBOFLOW_CONFIDENCE=0.35
VITE_AGGLUTINATION_THRESHOLD=0.12
```

## Notes

- `blood_type/` is kept for reference only; nothing in the React app reads it.
- Captures are no longer written to disk. The annotated frame is offered as a
  download instead of being saved to a `captures/` folder.
- Not a medical device — for demonstration and research only.
