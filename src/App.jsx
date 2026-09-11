/**
 * Blood Group Detector — a single-file React port of the Flask project that
 * used to live in blood_type/blood.py + blood_type/templates/blood_dashboard.html.
 *
 * Everything the Python server did now runs in the browser:
 *   - the webcam feed is a native <video> element (no MJPEG re-encoding, so the
 *     preview runs at the camera's real frame rate),
 *   - the Roboflow inference call goes straight to the hosted API,
 *   - the OpenCV agglutination pipeline (blur -> LAB/CLAHE -> adaptive threshold
 *     -> morphology -> contour + texture scoring) is reimplemented on canvas
 *     pixels below, including the dark-card layout fallback.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

/* ------------------------------------------------------------------ *
 * Configuration (was environment variables / setx on the Python side)
 * ------------------------------------------------------------------ */
const ROBOFLOW_API_KEY = import.meta.env.VITE_ROBOFLOW_API_KEY || 'vgpD5WtFfk4QRRqu6833'
const MODEL_ID = import.meta.env.VITE_ROBOFLOW_MODEL_ID || 'blood-group-detection-4yvdx/1'
const ROBOFLOW_CONFIDENCE = Number(import.meta.env.VITE_ROBOFLOW_CONFIDENCE ?? 0.35)
const AGGLUTINATION_THRESHOLD = Number(import.meta.env.VITE_AGGLUTINATION_THRESHOLD ?? 0.12)

const REGION_KEYS = ['A', 'B', 'D']
const ROI_SIZE = 300
const CAPTURE_WIDTH = 1280
const CAPTURE_HEIGHT = 720
const TARGET_FPS = 60
const LIVE_MODEL_INTERVAL_MS = 1800 // how often live mode re-asks Roboflow for regions
const LIVE_LOCAL_INTERVAL_MS = 150 // how often the local agglutination pass re-runs
const FALLBACK_MAX_WIDTH = 640 // the card-layout fallback works on a downscaled copy

const CLASS_ALIASES = {
  A: 'A', 'ANTI-A': 'A', ANTIA: 'A',
  B: 'B', 'ANTI-B': 'B', ANTIB: 'B',
  D: 'D', 'ANTI-D': 'D', ANTID: 'D', RH: 'D', RHD: 'D',
}

/* ------------------------------------------------------------------ *
 * Generic image primitives (the cv2.* calls blood.py relied on)
 * ------------------------------------------------------------------ */

/** cv2.getGaussianKernel(ksize, 0) — sigma is derived from the kernel size. */
function gaussianKernel(ksize) {
  const sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
  const half = (ksize - 1) / 2
  const kernel = new Float32Array(ksize)
  let sum = 0
  for (let i = 0; i < ksize; i += 1) {
    const x = i - half
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma))
    sum += kernel[i]
  }
  for (let i = 0; i < ksize; i += 1) kernel[i] /= sum
  return kernel
}

const KERNEL_CACHE = new Map()
function cachedKernel(ksize) {
  let kernel = KERNEL_CACHE.get(ksize)
  if (!kernel) {
    kernel = gaussianKernel(ksize)
    KERNEL_CACHE.set(ksize, kernel)
  }
  return kernel
}

/** BORDER_REFLECT_101 / BORDER_REPLICATE index mapping. */
function borderIndex(i, n, replicate) {
  if (i >= 0 && i < n) return i
  if (replicate) return i < 0 ? 0 : n - 1
  let v = i
  while (v < 0 || v >= n) {
    if (v < 0) v = -v
    if (v >= n) v = 2 * (n - 1) - v
  }
  return v
}

/** cv2.GaussianBlur on one channel — separable, with an interior fast path. */
function gaussianBlurChannel(src, w, h, ksize, replicate = false) {
  const kernel = cachedKernel(ksize)
  const half = (ksize - 1) / 2
  const tmp = new Float32Array(w * h)
  const out = new Float32Array(w * h)
  for (let y = 0; y < h; y += 1) {
    const row = y * w
    for (let x = 0; x < w; x += 1) {
      let acc = 0
      if (x >= half && x < w - half) {
        for (let t = 0; t < ksize; t += 1) acc += kernel[t] * src[row + x + t - half]
      } else {
        for (let t = 0; t < ksize; t += 1) acc += kernel[t] * src[row + borderIndex(x + t - half, w, replicate)]
      }
      tmp[row + x] = acc
    }
  }
  for (let y = 0; y < h; y += 1) {
    const row = y * w
    const interior = y >= half && y < h - half
    for (let x = 0; x < w; x += 1) {
      let acc = 0
      if (interior) {
        for (let t = 0; t < ksize; t += 1) acc += kernel[t] * tmp[(y + t - half) * w + x]
      } else {
        for (let t = 0; t < ksize; t += 1) acc += kernel[t] * tmp[borderIndex(y + t - half, h, replicate) * w + x]
      }
      out[row + x] = acc
    }
  }
  return out
}

/* --- sRGB <-> CIE L*a*b* (cv2.COLOR_BGR2LAB treats 8-bit input as sRGB) ---
 * Math.pow/cbrt dominate this stage, so each transfer function is tabulated
 * once and read back with linear interpolation. That keeps live mode cheap
 * while staying well inside 8-bit rounding of the OpenCV original. */
const LAB_XN = 0.950456
const LAB_ZN = 1.088754
const LUT_STEPS = 4096

function buildLut(from, to, fn) {
  const table = new Float64Array(LUT_STEPS + 1)
  for (let i = 0; i <= LUT_STEPS; i += 1) table[i] = fn(from + ((to - from) * i) / LUT_STEPS)
  return { table, from, scale: LUT_STEPS / (to - from) }
}

function sampleLut(lut, value) {
  const position = (value - lut.from) * lut.scale
  if (position <= 0) return lut.table[0]
  if (position >= LUT_STEPS) return lut.table[LUT_STEPS]
  const index = position | 0
  const fraction = position - index
  const low = lut.table[index]
  return low + (lut.table[index + 1] - low) * fraction
}

const SRGB_TO_LINEAR = buildLut(0, 255, (value) => {
  const c = value / 255
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
})

const LINEAR_TO_SRGB = buildLut(0, 1, (c) =>
  c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055,
)

const LAB_F = buildLut(0, 1.2, (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116))

function srgbToLinear(value) {
  return sampleLut(SRGB_TO_LINEAR, value)
}

function linearToSrgb(c) {
  const byte = Math.round(sampleLut(LINEAR_TO_SRGB, c) * 255)
  return byte < 0 ? 0 : byte > 255 ? 255 : byte
}

function labF(t) {
  return sampleLut(LAB_F, t)
}

function labFInverse(t) {
  return t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787
}

/**
 * cv2.createCLAHE(clipLimit, tileGridSize).apply(lightness)
 * Per-tile clipped histogram equalisation, bilinearly blended across tiles.
 */
function clahe(src, w, h, clipLimit = 2.0, tiles = 8) {
  const edges = (size) => {
    const values = new Int32Array(tiles + 1)
    for (let i = 0; i <= tiles; i += 1) values[i] = Math.round((i * size) / tiles)
    return values
  }
  const xs = edges(w)
  const ys = edges(h)
  const luts = []
  const hist = new Int32Array(256)

  for (let ty = 0; ty < tiles; ty += 1) {
    for (let tx = 0; tx < tiles; tx += 1) {
      hist.fill(0)
      for (let y = ys[ty]; y < ys[ty + 1]; y += 1) {
        const row = y * w
        for (let x = xs[tx]; x < xs[tx + 1]; x += 1) hist[src[row + x]] += 1
      }
      const area = Math.max(1, (ys[ty + 1] - ys[ty]) * (xs[tx + 1] - xs[tx]))
      if (clipLimit > 0) {
        const limit = Math.max(1, Math.round((clipLimit * area) / 256))
        let clipped = 0
        for (let i = 0; i < 256; i += 1) {
          if (hist[i] > limit) {
            clipped += hist[i] - limit
            hist[i] = limit
          }
        }
        // Redistribute the clipped mass the way OpenCV's CLAHE does.
        const batch = Math.floor(clipped / 256)
        let residual = clipped - batch * 256
        for (let i = 0; i < 256; i += 1) hist[i] += batch
        if (residual > 0) {
          const step = Math.max(Math.floor(256 / residual), 1)
          for (let i = 0; i < 256 && residual > 0; i += step, residual -= 1) hist[i] += 1
        }
      }
      const lut = new Uint8Array(256)
      const scale = 255 / area
      let sum = 0
      for (let i = 0; i < 256; i += 1) {
        sum += hist[i]
        const value = Math.round(sum * scale)
        lut[i] = value > 255 ? 255 : value
      }
      luts.push(lut)
    }
  }

  const out = new Uint8Array(w * h)
  const tileW = w / tiles
  const tileH = h / tiles
  for (let y = 0; y < h; y += 1) {
    const fy = y / tileH - 0.5
    const floorY = Math.floor(fy)
    const ya = fy - floorY
    const top = Math.min(Math.max(floorY, 0), tiles - 1) * tiles
    const bottom = Math.min(Math.max(floorY + 1, 0), tiles - 1) * tiles
    const row = y * w
    for (let x = 0; x < w; x += 1) {
      const fx = x / tileW - 0.5
      const floorX = Math.floor(fx)
      const xa = fx - floorX
      const left = Math.min(Math.max(floorX, 0), tiles - 1)
      const right = Math.min(Math.max(floorX + 1, 0), tiles - 1)
      const value = src[row + x]
      const topLeft = luts[top + left][value]
      const topRight = luts[top + right][value]
      const bottomLeft = luts[bottom + left][value]
      const bottomRight = luts[bottom + right][value]
      const upper = topLeft + (topRight - topLeft) * xa
      const lower = bottomLeft + (bottomRight - bottomLeft) * xa
      out[row + x] = Math.round(upper + (lower - upper) * ya)
    }
  }
  return out
}

/** cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block, C) */
function adaptiveThresholdInverse(gray, w, h, blockSize, delta) {
  const mean = gaussianBlurChannel(gray, w, h, blockSize, true)
  const out = new Uint8Array(w * h)
  for (let i = 0; i < out.length; i += 1) {
    out[i] = gray[i] > Math.round(mean[i]) - delta ? 0 : 255
  }
  return out
}

/** 3x3 erode/dilate, separable, with OpenCV's "unconstrained" default border. */
function morph3(src, w, h, dilate) {
  const pick = dilate ? Math.max : Math.min
  const outside = dilate ? 0 : 255
  const tmp = new Uint8Array(w * h)
  const out = new Uint8Array(w * h)
  for (let y = 0; y < h; y += 1) {
    const row = y * w
    for (let x = 0; x < w; x += 1) {
      const left = x > 0 ? src[row + x - 1] : outside
      const right = x < w - 1 ? src[row + x + 1] : outside
      tmp[row + x] = pick(left, src[row + x], right)
    }
  }
  for (let y = 0; y < h; y += 1) {
    const row = y * w
    for (let x = 0; x < w; x += 1) {
      const up = y > 0 ? tmp[row - w + x] : outside
      const down = y < h - 1 ? tmp[row + w + x] : outside
      out[row + x] = pick(up, tmp[row + x], down)
    }
  }
  return out
}

/**
 * Sliding-window extremum over one line, via a monotonic deque: O(n) instead of
 * O(n * kernel). The line is padded with the neutral border value so the result
 * matches OpenCV's default morphology border.
 */
function slidingExtreme(line, n, kernel, dilate, outside, out) {
  const half = kernel >> 1
  const span = n + 2 * half
  const padded = new Uint8Array(span).fill(outside)
  padded.set(line.subarray(0, n), half)
  const deque = new Int32Array(span)
  let head = 0
  let tail = -1
  for (let i = 0; i < span; i += 1) {
    const value = padded[i]
    while (tail >= head && (dilate ? padded[deque[tail]] <= value : padded[deque[tail]] >= value)) tail -= 1
    deque[(tail += 1)] = i
    const target = i - 2 * half
    if (target >= 0) {
      while (deque[head] < target) head += 1
      out[target] = padded[deque[head]]
    }
  }
}

/** Rectangular structuring element erode/dilate (used by the card fallback). */
function rectMorph(src, w, h, kw, kh, dilate) {
  const outside = dilate ? 0 : 255
  const tmp = new Uint8Array(w * h)
  const out = new Uint8Array(w * h)
  const lineIn = new Uint8Array(Math.max(w, h))
  const lineOut = new Uint8Array(Math.max(w, h))
  for (let y = 0; y < h; y += 1) {
    const row = y * w
    lineIn.set(src.subarray(row, row + w))
    slidingExtreme(lineIn, w, kw, dilate, outside, lineOut)
    tmp.set(lineOut.subarray(0, w), row)
  }
  for (let x = 0; x < w; x += 1) {
    for (let y = 0; y < h; y += 1) lineIn[y] = tmp[y * w + x]
    slidingExtreme(lineIn, h, kh, dilate, outside, lineOut)
    for (let y = 0; y < h; y += 1) out[y * w + x] = lineOut[y]
  }
  return out
}

/* Clockwise 8-neighbour offsets: E, SE, S, SW, W, NW, N, NE. */
const TRACE_DX = [1, 1, 0, -1, -1, -1, 0, 1]
const TRACE_DY = [0, 1, 1, 1, 0, -1, -1, -1]

/**
 * cv2.contourArea() of a blob's outer border: Moore-neighbour boundary tracing
 * plus the shoelace formula, which is what OpenCV measures. Using the raw pixel
 * count instead would over-report every clump by roughly half its perimeter and
 * nudge the agglutination score upward.
 */
function outerContourArea(mask, w, h, startIndex) {
  const startY = (startIndex / w) | 0
  const startX = startIndex - startY * w
  let x = startX
  let y = startY
  let backtrack = 4 // the row-major scan arrives from the west
  let firstDirection = -1
  const xs = []
  const ys = []
  const maxSteps = 4 * w * h + 16
  for (let step = 0; step < maxSteps; step += 1) {
    xs.push(x)
    ys.push(y)
    let direction = -1
    for (let k = 1; k <= 8; k += 1) {
      const candidate = (backtrack + k) % 8
      const nx = x + TRACE_DX[candidate]
      const ny = y + TRACE_DY[candidate]
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
      if (!mask[ny * w + nx]) continue
      direction = candidate
      break
    }
    if (direction < 0) break // isolated pixel
    if (x === startX && y === startY) {
      if (firstDirection < 0) {
        firstDirection = direction
      } else if (direction === firstDirection) {
        xs.pop() // Jacob's stopping criterion: the border is closed
        ys.pop()
        break
      }
    }
    backtrack = (direction + 4) % 8
    x += TRACE_DX[direction]
    y += TRACE_DY[direction]
  }
  let sum = 0
  for (let i = 0; i < xs.length; i += 1) {
    const j = (i + 1) % xs.length
    sum += xs[i] * ys[j] - xs[j] * ys[i]
  }
  return Math.abs(sum) / 2
}

/**
 * cv2.findContours(..., RETR_EXTERNAL) reduced to what blood.py actually used:
 * one entry per 8-connected blob, with its contour area, pixel count and box.
 */
function connectedComponents(mask, w, h) {
  const seen = new Uint8Array(w * h)
  const stack = new Int32Array(w * h)
  const blobs = []
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || seen[start]) continue
    let top = -1
    stack[(top += 1)] = start
    seen[start] = 1
    let area = 0
    let minX = w
    let maxX = 0
    let minY = h
    let maxY = 0
    while (top >= 0) {
      const index = stack[top]
      top -= 1
      const y = (index / w) | 0
      const x = index - y * w
      area += 1
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      for (let dy = -1; dy <= 1; dy += 1) {
        const ny = y + dy
        if (ny < 0 || ny >= h) continue
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx
          if (nx < 0 || nx >= w) continue
          const neighbour = ny * w + nx
          if (mask[neighbour] && !seen[neighbour]) {
            seen[neighbour] = 1
            stack[(top += 1)] = neighbour
          }
        }
      }
    }
    blobs.push({
      area,
      contourArea: outerContourArea(mask, w, h, start),
      x: minX,
      y: minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1,
    })
  }
  return blobs
}

/** cv2.Laplacian(gray, CV_64F).var() with the default reflect-101 border. */
function laplacianVariance(gray, w, h) {
  const n = w * h
  let sum = 0
  let sumSquares = 0
  for (let y = 0; y < h; y += 1) {
    const row = y * w
    const up = borderIndex(y - 1, h, false) * w
    const down = borderIndex(y + 1, h, false) * w
    for (let x = 0; x < w; x += 1) {
      const left = x > 0 ? gray[row + x - 1] : gray[row + 1]
      const right = x < w - 1 ? gray[row + x + 1] : gray[row + w - 2]
      const value = gray[up + x] + gray[down + x] + left + right - 4 * gray[row + x]
      sum += value
      sumSquares += value * value
    }
  }
  const mean = sum / n
  return sumSquares / n - mean * mean
}

/** Contiguous runs of truthy values (replaces blood.py's np.diff run finder). */
function findRuns(flags, length) {
  const runs = []
  let start = -1
  for (let i = 0; i <= length; i += 1) {
    const on = i < length && flags[i]
    if (on && start < 0) start = i
    else if (!on && start >= 0) {
      runs.push([start, i])
      start = -1
    }
  }
  return runs
}

function imageDataToGray(image) {
  const { width: w, height: h, data } = image
  const gray = new Uint8Array(w * h)
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    gray[i] = (0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2] + 0.5) | 0
  }
  return gray
}

/* ------------------------------------------------------------------ *
 * The blood-card analysis itself (preprocess_roi + detect_agglutination)
 * ------------------------------------------------------------------ */

/**
 * preprocess_roi(): blur, push the image through LAB so CLAHE can lift only the
 * lightness channel, then come back to RGB and flatten to grey.
 */
function preprocessToGray(image) {
  const { width: w, height: h, data } = image
  const n = w * h
  const red = new Float32Array(n)
  const green = new Float32Array(n)
  const blue = new Float32Array(n)
  for (let i = 0, p = 0; i < n; i += 1, p += 4) {
    red[i] = data[p]
    green[i] = data[p + 1]
    blue[i] = data[p + 2]
  }

  const redBlur = gaussianBlurChannel(red, w, h, 5)
  const greenBlur = gaussianBlurChannel(green, w, h, 5)
  const blueBlur = gaussianBlurChannel(blue, w, h, 5)

  const lightness = new Uint8Array(n)
  const deltaA = new Float32Array(n)
  const deltaB = new Float32Array(n)
  for (let i = 0; i < n; i += 1) {
    const r = srgbToLinear(redBlur[i])
    const g = srgbToLinear(greenBlur[i])
    const b = srgbToLinear(blueBlur[i])
    const fx = labF((0.412453 * r + 0.35758 * g + 0.180423 * b) / LAB_XN)
    const fy = labF(0.212671 * r + 0.71516 * g + 0.072169 * b)
    const fz = labF((0.019334 * r + 0.119193 * g + 0.950227 * b) / LAB_ZN)
    const l = 116 * fy - 16
    const quantised = Math.round((l * 255) / 100)
    lightness[i] = quantised < 0 ? 0 : quantised > 255 ? 255 : quantised
    deltaA[i] = fx - fy
    deltaB[i] = fy - fz
  }

  const enhanced = clahe(lightness, w, h, 2.0, 8)
  const gray = new Uint8Array(n)
  for (let i = 0; i < n; i += 1) {
    const l = (enhanced[i] * 100) / 255
    const fy = (l + 16) / 116
    const y = labFInverse(fy)
    const x = labFInverse(fy + deltaA[i]) * LAB_XN
    const z = labFInverse(fy - deltaB[i]) * LAB_ZN
    const r = linearToSrgb(3.240479 * x - 1.53715 * y - 0.498535 * z)
    const g = linearToSrgb(-0.969256 * x + 1.875991 * y + 0.041556 * z)
    const b = linearToSrgb(0.055648 * x - 0.204043 * y + 1.057311 * z)
    gray[i] = (0.299 * r + 0.587 * g + 0.114 * b + 0.5) | 0
  }

  // cv2.GaussianBlur(gray, (3, 3), 0) — the same buffer feeds both the
  // threshold and the Laplacian texture measure, exactly as in blood.py.
  const blurred = gaussianBlurChannel(gray, w, h, 3)
  const smoothed = new Uint8Array(n)
  for (let i = 0; i < n; i += 1) {
    const value = Math.round(blurred[i])
    smoothed[i] = value < 0 ? 0 : value > 255 ? 255 : value
  }
  return smoothed
}

/** detect_agglutination(): clump ratio + texture + clump count -> score. */
function detectAgglutination(image) {
  const w = image.width
  const h = image.height
  const gray = preprocessToGray(image)

  let mask = adaptiveThresholdInverse(gray, w, h, 21, 4)
  mask = morph3(morph3(mask, w, h, false), w, h, true) // MORPH_OPEN
  mask = morph3(morph3(mask, w, h, true), w, h, false) // MORPH_CLOSE

  const significant = connectedComponents(mask, w, h).filter((blob) => blob.contourArea > 80)
  const clumpArea = significant.reduce((total, blob) => total + blob.contourArea, 0)
  const clumpRatio = clumpArea / (w * h)
  const texture = laplacianVariance(gray, w, h)
  const score =
    0.6 * Math.min(clumpRatio / 0.3, 1) +
    0.25 * Math.min(texture / 1000, 1) +
    0.15 * Math.min(significant.length / 30, 1)

  return {
    positive: score >= AGGLUTINATION_THRESHOLD,
    score,
    clumps: significant.length,
    clumpRatio,
    texture,
    mask,
    width: w,
    height: h,
  }
}

/** determine_blood_group(anti_a, anti_b, anti_d) */
function determineBloodGroup(antiA, antiB, antiD) {
  let abo
  if (antiA && !antiB) abo = 'A'
  else if (!antiA && antiB) abo = 'B'
  else if (antiA && antiB) abo = 'AB'
  else abo = 'O'
  return abo + (antiD ? '+' : '-')
}

/* ------------------------------------------------------------------ *
 * Region discovery — Roboflow first, dark-card geometry as a fallback
 * ------------------------------------------------------------------ */

/** extract_rois(): keep the most confident A / B / D box per antigen. */
function extractPredictionBoxes(frameWidth, frameHeight, result) {
  const detected = {}
  for (const prediction of result?.predictions ?? []) {
    const rawName = String(prediction.class ?? '').toUpperCase().replace(/_/g, '-').replace(/ /g, '-')
    const name = CLASS_ALIASES[rawName] ?? rawName
    const confidence = Number(prediction.confidence ?? 0)
    if (!REGION_KEYS.includes(name) || confidence < ROBOFLOW_CONFIDENCE) continue
    const x = Math.trunc(prediction.x)
    const y = Math.trunc(prediction.y)
    const width = Math.trunc(prediction.width)
    const height = Math.trunc(prediction.height)
    const box = [
      Math.max(0, x - Math.floor(width / 2)),
      Math.max(0, y - Math.floor(height / 2)),
      Math.min(frameWidth, x + Math.floor(width / 2)),
      Math.min(frameHeight, y + Math.floor(height / 2)),
    ]
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) continue
    if (!detected[name] || confidence > detected[name].confidence) {
      detected[name] = { box, confidence, source: 'roboflow' }
    }
  }
  return detected
}

/**
 * extract_card_rois(): find the wide dark test card and split it into three
 * wells left-to-right. Runs on a downscaled copy so it stays cheap in live mode;
 * every threshold in the original is relative to the frame size, so the boxes
 * simply scale back up.
 */
function extractCardBoxes(gray, w, h, scale) {
  const rawDark = new Uint8Array(w * h)
  for (let i = 0; i < rawDark.length; i += 1) rawDark[i] = gray[i] <= 125 ? 255 : 0

  // MORPH_CLOSE with a 21x11 rect, two iterations. Applying a flat rectangle
  // twice equals one pass with the dilated rectangle (21+20, 11+10), so this is
  // two sweeps rather than four.
  const dark = rectMorph(rectMorph(rawDark, w, h, 41, 21, true), w, h, 41, 21, false)

  const candidates = []
  for (const blob of connectedComponents(dark, w, h)) {
    const ratio = blob.width / Math.max(blob.height, 1)
    if (
      blob.width >= w * 0.45 && blob.width <= w * 0.95 &&
      blob.height >= h * 0.1 && blob.height <= h * 0.6 &&
      ratio >= 2 && ratio <= 6.5
    ) {
      candidates.push({ area: blob.width * blob.height, x: blob.x, y: blob.y, width: blob.width, height: blob.height })
    }
  }

  // A card sitting against a dark panel merges into one huge blob. Projection
  // profiles still expose the wide horizontal band and bridge across the wells.
  if (!candidates.length) {
    const rowActive = new Uint8Array(h)
    for (let y = 0; y < h; y += 1) {
      let on = 0
      const row = y * w
      for (let x = 0; x < w; x += 1) if (rawDark[row + x]) on += 1
      rowActive[y] = on / w > 0.55 ? 1 : 0
    }
    for (const [rowStart, rowEnd] of findRuns(rowActive, h)) {
      const bandHeight = rowEnd - rowStart
      if (bandHeight < h * 0.1 || bandHeight > h * 0.6) continue
      const columnFlags = new Uint8Array(w)
      for (let x = 0; x < w; x += 1) {
        let on = 0
        for (let y = rowStart; y < rowEnd; y += 1) if (rawDark[y * w + x]) on += 1
        columnFlags[x] = on / bandHeight > 0.42 ? 1 : 0
      }
      for (const [runStart, runEnd] of findRuns(columnFlags, w)) {
        if (runEnd - runStart < w * 0.04) columnFlags.fill(0, runStart, runEnd)
      }
      const strip = new Uint8Array(w)
      for (let x = 0; x < w; x += 1) strip[x] = columnFlags[x] ? 255 : 0
      // Bridge the bright wells, but not the gap to a neighbouring dark panel.
      const bridge = Math.max(11, Math.min(21, Math.floor(w / 80)))
      const closed = rectMorph(rectMorph(strip, w, 1, bridge, 1, true), w, 1, bridge, 1, false)
      for (const [columnStart, columnEnd] of findRuns(closed, w)) {
        const columnWidth = columnEnd - columnStart
        const ratio = columnWidth / Math.max(bandHeight, 1)
        if (columnWidth >= w * 0.45 && columnWidth <= w * 0.95 && ratio >= 2 && ratio <= 6.5) {
          candidates.push({
            area: columnWidth * bandHeight,
            x: columnStart,
            y: rowStart,
            width: columnWidth,
            height: bandHeight,
          })
        }
      }
    }
  }

  if (!candidates.length) return {}

  const card = candidates.reduce((best, item) => (item.area > best.area ? item : best))
  const cellWidth = card.width / 3
  const detected = {}
  REGION_KEYS.forEach((name, position) => {
    const cellStart = card.x + position * cellWidth
    // Trim each third so the neighbouring wells and card edges stay out.
    // Truncated, not rounded, to match the int() casts in the original.
    const box = [
      Math.round(Math.trunc(cellStart + cellWidth * 0.1) * scale),
      Math.round(Math.trunc(card.y + card.height * 0.1) * scale),
      Math.round(Math.trunc(cellStart + cellWidth * 0.9) * scale),
      Math.round(Math.trunc(card.y + card.height * 0.9) * scale),
    ]
    detected[name] = { box, confidence: 0, source: 'card-layout-fallback' }
  })
  return detected
}

/* ------------------------------------------------------------------ *
 * Roboflow hosted inference
 * ------------------------------------------------------------------ */
const INFERENCE_HOSTS = ['https://serverless.roboflow.com', 'https://detect.roboflow.com']

async function inferRoboflow(dataUrl, signal) {
  if (!ROBOFLOW_API_KEY) throw new Error('No Roboflow API key is configured.')
  const base64 = dataUrl.slice(dataUrl.indexOf(',') + 1)
  let lastError = null
  for (const host of INFERENCE_HOSTS) {
    try {
      const response = await fetch(`${host}/${MODEL_ID}?api_key=${encodeURIComponent(ROBOFLOW_API_KEY)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: base64,
        signal,
      })
      if (!response.ok) {
        const detail = await response.text().catch(() => '')
        throw new Error(`Roboflow responded ${response.status}. ${detail.slice(0, 180)}`.trim())
      }
      return await response.json()
    } catch (error) {
      if (error.name === 'AbortError') throw error
      lastError = error
    }
  }
  throw new Error(`Could not reach the Roboflow model. ${lastError?.message ?? ''}`.trim())
}

/* ------------------------------------------------------------------ *
 * Canvas plumbing (replaces the capture files blood.py wrote to disk)
 * ------------------------------------------------------------------ */
function makeCanvas(width, height) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return canvas
}

function context2d(canvas) {
  return canvas.getContext('2d', { willReadFrequently: true })
}

function cropToImageData(source, box, size) {
  const [x1, y1, x2, y2] = box
  const canvas = makeCanvas(size, size)
  const ctx = context2d(canvas)
  ctx.drawImage(source, x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1), 0, 0, size, size)
  return { image: ctx.getImageData(0, 0, size, size), canvas }
}

function maskToDataUrl(mask, width, height) {
  const canvas = makeCanvas(width, height)
  const ctx = context2d(canvas)
  const image = ctx.createImageData(width, height)
  for (let i = 0, p = 0; i < mask.length; i += 1, p += 4) {
    image.data[p] = mask[i]
    image.data[p + 1] = mask[i]
    image.data[p + 2] = mask[i]
    image.data[p + 3] = 255
  }
  ctx.putImageData(image, 0, 0)
  return canvas.toDataURL('image/png')
}

/** draw_result(): the same box + label styling the OpenCV annotation used. */
function drawRegionBox(ctx, name, box, positive, score, scale = 1) {
  const [x1, y1, x2, y2] = box
  const colour = positive ? 'rgb(130, 201, 64)' : 'rgb(255, 160, 91)'
  const lineWidth = Math.max(2, Math.round(3 * scale))
  ctx.lineWidth = lineWidth
  ctx.strokeStyle = colour
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
  const fontSize = Math.max(13, Math.round(18 * scale))
  ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`
  ctx.textBaseline = 'bottom'
  const label = `Anti-${name}: ${positive ? 'POSITIVE' : 'NEGATIVE'} (${score.toFixed(2)})`
  const textY = Math.max(y1 - 6, fontSize + 4)
  const metrics = ctx.measureText(label)
  ctx.fillStyle = 'rgba(5, 10, 20, 0.72)'
  ctx.fillRect(x1, textY - fontSize - 2, metrics.width + 10, fontSize + 8)
  ctx.fillStyle = colour
  ctx.fillText(label, x1 + 5, textY + 2)
}

/**
 * analyse_frame(): score the located wells, decide the group, and — for a real
 * capture — render the annotated frame plus per-well thumbnails.
 *
 * `options.only` scores a single well and reuses `options.previous` for the
 * other two. Live mode cycles through A, B and D that way, so one frame of work
 * costs ~50ms instead of ~150ms and the preview never stalls.
 */
function analyseRegions(frameCanvas, detected, withThumbnails, options = {}) {
  const { only = null, previous = null } = options
  const targets = only ? [only] : REGION_KEYS
  const regions = { ...(previous ?? {}) }

  for (const name of targets) {
    const found = detected[name]
    if (!found) continue
    const { image, canvas } = cropToImageData(frameCanvas, found.box, ROI_SIZE)
    const analysis = detectAgglutination(image)
    regions[name] = {
      positive: analysis.positive,
      status: analysis.positive ? 'POSITIVE' : 'NEGATIVE',
      score: Math.round(analysis.score * 1000) / 1000,
      clumps: analysis.clumps,
      clumpRatio: Math.round(analysis.clumpRatio * 1000) / 1000,
      texture: Math.round(analysis.texture),
      confidence: Math.round(found.confidence * 1000) / 1000,
      source: found.source,
      box: found.box,
      roiImage: withThumbnails ? canvas.toDataURL('image/jpeg', 0.88) : null,
      processedImage: withThumbnails ? maskToDataUrl(analysis.mask, analysis.width, analysis.height) : null,
    }
  }
  if (REGION_KEYS.some((name) => !regions[name])) return null

  let annotatedImage = null
  if (withThumbnails) {
    const annotated = makeCanvas(frameCanvas.width, frameCanvas.height)
    const annotatedCtx = annotated.getContext('2d')
    annotatedCtx.drawImage(frameCanvas, 0, 0)
    const scale = frameCanvas.width / CAPTURE_WIDTH
    for (const name of REGION_KEYS) {
      const region = regions[name]
      drawRegionBox(annotatedCtx, name, region.box, region.positive, region.score, scale)
    }
    annotatedImage = annotated.toDataURL('image/jpeg', 0.9)
  }

  return {
    ok: true,
    bloodGroup: determineBloodGroup(regions.A.positive, regions.B.positive, regions.D.positive),
    regions,
    detectionMode: REGION_KEYS.every((name) => regions[name].source === 'roboflow')
      ? 'Roboflow model'
      : 'Roboflow model + card layout',
    annotatedImage,
    capturedAt: new Date().toISOString(),
  }
}

function describeMissing(result, missing) {
  const observed = (result?.predictions ?? []).map((prediction) => ({
    label: String(prediction.class ?? 'unknown'),
    confidence: Math.round(Number(prediction.confidence ?? 0) * 1000) / 1000,
  }))
  const text = observed.map((item) => `${item.label} (${(item.confidence * 100).toFixed(1)}%)`).join(', ') || 'none'
  return {
    message: `The model did not detect all A, B and D regions. It returned: ${text}.`,
    missing,
    hint: 'Show one physical card directly to the camera and fill the frame with it; do not film the card off a monitor.',
  }
}

/* ------------------------------------------------------------------ *
 * Camera selection — "auto available camera"
 * ------------------------------------------------------------------ */
const EXTERNAL_HINT = /(usb|uvc|external|logi|logitech|c\d{3}|brio|cam\s?link|capture|webcam)/i
const INTERNAL_HINT = /(integrated|built[\s-]?in|internal|facetime|laptop|ir\b|infrared)/i
const VIRTUAL_HINT = /(virtual|obs|snap|droidcam|manycam|xsplit|epoccam|iriun|splitcam)/i
const FRONT_HINT = /(front|user|selfie|facetime)/i
const BACK_HINT = /(back|rear|environment|world)/i

/**
 * A short tag for what kind of camera this is. Order matters: "Integrated
 * Webcam" contains both "integrated" and "webcam", and it is a built-in camera,
 * so the internal test has to win.
 */
function cameraKind(device) {
  const label = device.label || ''
  if (!label) return ''
  if (VIRTUAL_HINT.test(label)) return 'virtual'
  if (INTERNAL_HINT.test(label)) return 'built-in'
  if (BACK_HINT.test(label)) return 'back'
  if (FRONT_HINT.test(label)) return 'front'
  if (EXTERNAL_HINT.test(label)) return 'USB / external'
  return ''
}

/* The camera the user last chose by hand, so a reload does not quietly hand
   them a different one. */
const CAMERA_STORAGE_KEY = 'blood-group-detector.camera'

function rememberCamera(deviceId) {
  try {
    if (deviceId) localStorage.setItem(CAMERA_STORAGE_KEY, deviceId)
    else localStorage.removeItem(CAMERA_STORAGE_KEY)
  } catch {
    // Private mode / blocked storage: the choice just will not outlive the tab.
  }
}

function recallCamera() {
  try {
    return localStorage.getItem(CAMERA_STORAGE_KEY) || ''
  } catch {
    return ''
  }
}

/**
 * Rank inputs for the automatic pick only — a real USB card camera first, a
 * phone's back camera next, a laptop lid camera after that. An explicit choice
 * by the user always overrides this.
 */
const KIND_SCORE = { 'USB / external': 3, back: 2, '': 1, front: 0, 'built-in': 0, virtual: -2 }

function cameraPreference(devices) {
  return devices
    .map((device, index) => {
      const kind = cameraKind(device)
      // With no labels at all we cannot tell them apart; index 1 is the USB slot
      // on most machines, which is what the Python build defaulted to.
      const score = !device.label && devices.length > 1 && index > 0 ? 2 : KIND_SCORE[kind] ?? 1
      return { device, score, index }
    })
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((entry) => entry.device)
}

/**
 * Browsers only reveal real device labels once camera permission is granted, so
 * fall back to a position-based name that still tells the two common cases
 * apart.
 */
function cameraLabel(device, index) {
  const name = device.label || (index === 0 ? 'Camera 1 (usually built-in)' : `Camera ${index + 1} (USB/external)`)
  const kind = cameraKind(device)
  return kind && !new RegExp(kind.split(' ')[0], 'i').test(name) ? `${name} — ${kind}` : name
}

/**
 * Wishes only — every field is `ideal`. A mandatory `min` frame rate makes
 * getUserMedia throw OverconstrainedError on any camera that runs slower (many
 * webcams drop to 15fps in dim light), which would hide that camera completely.
 */
function videoConstraints(deviceId) {
  return {
    deviceId: deviceId ? { exact: deviceId } : undefined,
    width: { ideal: CAPTURE_WIDTH },
    height: { ideal: CAPTURE_HEIGHT },
    frameRate: { ideal: TARGET_FPS },
  }
}

/** Last resort: take the camera on any terms rather than not at all. */
function relaxedConstraints(deviceId) {
  return deviceId ? { deviceId: { exact: deviceId } } : true
}

/* ------------------------------------------------------------------ *
 * Dashboard
 * ------------------------------------------------------------------ */
export default function App() {
  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const streamRef = useRef(null)
  const frameCanvasRef = useRef(null)
  const generationRef = useRef(0)
  const liveRef = useRef({
    on: false,
    detected: null,
    regions: null,
    modelBusy: false,
    lastModel: 0,
    nextLocal: 0,
    lastPublish: 0,
    cursor: 0,
  })

  const [devices, setDevices] = useState([])
  const [activeDeviceId, setActiveDeviceId] = useState('')
  // true once the live camera is the user's own pick rather than the auto scan
  const [pinned, setPinned] = useState(false)
  const [track, setTrack] = useState(null)
  const [aspect, setAspect] = useState('16 / 9')
  const [fps, setFps] = useState(0)
  const [analysisMs, setAnalysisMs] = useState(0)
  const [status, setStatus] = useState({ kind: 'info', text: 'Looking for an available camera…' })
  const [busy, setBusy] = useState(false)
  const [live, setLive] = useState(false)
  const [result, setResult] = useState(null)
  const [failure, setFailure] = useState(null)

  const say = useCallback((text, kind = 'info') => setStatus({ kind, text }), [])

  const listCameras = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return []
    const all = await navigator.mediaDevices.enumerateDevices()
    const inputs = all.filter((device) => device.kind === 'videoinput')
    setDevices(inputs)
    return inputs
  }, [])

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((item) => item.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
  }, [])

  /** Open one specific camera and wait until it actually delivers frames. */
  const attach = useCallback(async (deviceId) => {
    const generation = (generationRef.current += 1)
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints(deviceId), audio: false })
    } catch (error) {
      // A resolution or frame-rate wish is never worth losing the camera over.
      if (error.name !== 'OverconstrainedError' && error.name !== 'NotReadableError') throw error
      stream = await navigator.mediaDevices.getUserMedia({ video: relaxedConstraints(deviceId), audio: false })
    }
    if (generation !== generationRef.current) {
      stream.getTracks().forEach((item) => item.stop())
      return null
    }
    streamRef.current?.getTracks().forEach((item) => item.stop())
    streamRef.current = stream
    const video = videoRef.current
    if (!video) {
      stream.getTracks().forEach((item) => item.stop())
      return null
    }
    video.srcObject = stream
    await video.play().catch(() => {})
    await new Promise((resolve) => {
      if (video.videoWidth) {
        resolve()
        return
      }
      const done = () => {
        video.removeEventListener('loadeddata', done)
        resolve()
      }
      video.addEventListener('loadeddata', done)
      setTimeout(done, 4000) // phone cameras can take a beat to hand over the first frame
    })
    if (!video.videoWidth) throw new Error('The camera opened but produced no frames.')
    const videoTrack = stream.getVideoTracks()[0]
    const settings = videoTrack.getSettings()
    setTrack({ label: videoTrack.label, settings })
    // Not every browser reports deviceId back in getSettings(), so keep the id we
    // asked for — the picker needs it to show which camera is live.
    setActiveDeviceId(settings.deviceId || deviceId || '')
    // Show the frame exactly as it is analysed: the panel takes the camera's own
    // aspect ratio instead of cropping a 4:3 sensor into a 16:9 box.
    setAspect(`${video.videoWidth} / ${video.videoHeight}`)
    return videoTrack
  }, [])

  /**
   * The Python build probed camera indices until one returned a frame; this is
   * the browser equivalent, preferring an external camera the way
   * CAMERA_INDEX=1 used to.
   */
  const describeCamera = useCallback((videoTrack, prefix) => {
    const settings = videoTrack.getSettings()
    const size = settings.width ? ` at ${settings.width}x${settings.height}` : ''
    const rate = settings.frameRate ? ` @ ${Math.round(settings.frameRate)}fps` : ''
    say(`${prefix} ${videoTrack.label || 'camera'}${size}${rate}. Hold the test card in view.`, 'ok')
  }, [say])

  const autoSelect = useCallback(
    async ({ useSaved = true } = {}) => {
      if (!navigator.mediaDevices?.getUserMedia) {
        say('This browser cannot access cameras. Use Chrome or Edge over http://localhost or https.', 'error')
        return
      }
      say(useSaved ? 'Starting the camera…' : 'Looking for an available camera…')
      try {
        // A first generic open grants permission, which is what makes device
        // labels (and therefore any preference at all) readable.
        let inputs = await listCameras()
        if (!inputs.some((device) => device.label)) {
          await attach(undefined)
          inputs = await listCameras()
        }

        // The user's own choice comes first and is never second-guessed.
        const saved = useSaved ? recallCamera() : ''
        let savedFailed = false
        if (saved && inputs.some((device) => device.deviceId === saved)) {
          try {
            const videoTrack = await attach(saved)
            if (!videoTrack) return
            setPinned(true)
            describeCamera(videoTrack, 'Using your chosen camera:')
            return
          } catch {
            savedFailed = true // busy or blocked — say so once a fallback is up
          }
        }

        for (const device of cameraPreference(inputs)) {
          try {
            const videoTrack = await attach(device.deviceId)
            if (!videoTrack) return
            setPinned(false)
            describeCamera(
              videoTrack,
              savedFailed ? 'Your chosen camera would not open (in use?) — falling back to' : 'Auto-selected',
            )
            return
          } catch {
            // Busy or blocked camera — fall through to the next candidate.
          }
        }
        if (!streamRef.current) throw new Error('No camera could be opened. Close any app already using it and retry.')
        setPinned(false)
        say('Camera ready. Hold the test card in view.', 'ok')
      } catch (error) {
        say(error.message || 'Camera access was blocked.', 'error')
      }
    },
    [attach, describeCamera, listCameras, say],
  )

  /** An explicit pick by the user: it is opened, and it is remembered. */
  const selectCamera = useCallback(
    async (deviceId) => {
      if (!deviceId) return
      setLive(false)
      say('Switching camera…')
      try {
        const videoTrack = await attach(deviceId)
        if (!videoTrack) return
        rememberCamera(deviceId)
        setPinned(true)
        describeCamera(videoTrack, 'Now using')
      } catch (error) {
        say(`${error.message || 'That camera could not be opened.'} It may be in use by another app.`, 'error')
      }
    },
    [attach, describeCamera, say],
  )

  /** Hand the choice back to the automatic scan. */
  const useAutoCamera = useCallback(async () => {
    rememberCamera('')
    setPinned(false)
    setLive(false)
    await autoSelect({ useSaved: false })
  }, [autoSelect])

  useEffect(() => {
    // Deferred by a tick so the first status update lands outside the effect body.
    const timer = setTimeout(autoSelect, 0)
    return () => {
      clearTimeout(timer)
      generationRef.current += 1
      stopStream()
    }
  }, [autoSelect, stopStream])

  // Re-scan when a camera is plugged in or unplugged, and recover if the one in
  // use disappears.
  useEffect(() => {
    const media = navigator.mediaDevices
    if (!media?.addEventListener) return undefined
    const onChange = async () => {
      const inputs = await listCameras()
      const stillThere = inputs.some((device) => device.deviceId === activeDeviceId)
      const alive = streamRef.current?.getVideoTracks()[0]?.readyState === 'live'
      const saved = recallCamera()

      // The camera the user chose has come back (replugged) — return to it.
      if (saved && saved !== activeDeviceId && inputs.some((device) => device.deviceId === saved)) {
        setLive(false)
        await selectCamera(saved)
        return
      }
      if (stillThere && alive) return // nothing to do; never override a working choice
      if (saved && saved === activeDeviceId) {
        say('Your chosen camera was disconnected. Looking for another one…', 'error')
      }
      setLive(false)
      await autoSelect()
    }
    media.addEventListener('devicechange', onChange)
    return () => media.removeEventListener('devicechange', onChange)
  }, [activeDeviceId, autoSelect, listCameras, selectCamera, say])

  // Real frame-rate meter: counts frames the camera actually presented.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return undefined
    let stopped = false
    let handle = 0
    let usedVideoCallback = false
    let frames = 0
    let last = performance.now()
    const measure = () => {
      frames += 1
      const now = performance.now()
      if (now - last >= 500) {
        setFps(Math.round((frames * 1000) / (now - last)))
        frames = 0
        last = now
      }
      if (!stopped) schedule()
    }
    const schedule = () => {
      if (video.requestVideoFrameCallback) {
        usedVideoCallback = true
        handle = video.requestVideoFrameCallback(measure)
      } else {
        usedVideoCallback = false
        handle = requestAnimationFrame(measure)
      }
    }
    schedule()
    return () => {
      stopped = true
      if (usedVideoCallback) video.cancelVideoFrameCallback?.(handle)
      else cancelAnimationFrame(handle)
    }
  }, [])

  /** Grab the current video frame into a reusable canvas. */
  const grabFrame = useCallback(() => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return null
    let canvas = frameCanvasRef.current
    if (!canvas) {
      canvas = makeCanvas(video.videoWidth, video.videoHeight)
      frameCanvasRef.current = canvas
    }
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }
    context2d(canvas).drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas
  }, [])

  /** Roboflow regions, with the dark-card geometry as a fallback. */
  const locateRegions = useCallback(async (frameCanvas, signal) => {
    const dataUrl = frameCanvas.toDataURL('image/jpeg', 0.9)
    const inference = await inferRoboflow(dataUrl, signal)
    const detected = extractPredictionBoxes(frameCanvas.width, frameCanvas.height, inference)
    const missing = REGION_KEYS.filter((name) => !detected[name])
    if (!missing.length) return { detected, inference, dataUrl }

    const scale = frameCanvas.width > FALLBACK_MAX_WIDTH ? FALLBACK_MAX_WIDTH / frameCanvas.width : 1
    const smallWidth = Math.max(1, Math.round(frameCanvas.width * scale))
    const smallHeight = Math.max(1, Math.round(frameCanvas.height * scale))
    const small = makeCanvas(smallWidth, smallHeight)
    const smallCtx = context2d(small)
    smallCtx.drawImage(frameCanvas, 0, 0, smallWidth, smallHeight)
    const gray = imageDataToGray(smallCtx.getImageData(0, 0, smallWidth, smallHeight))
    const fallback = extractCardBoxes(gray, smallWidth, smallHeight, 1 / scale)
    for (const name of missing) {
      if (fallback[name]) detected[name] = fallback[name]
    }
    const stillMissing = REGION_KEYS.filter((name) => !detected[name])
    return { detected, inference, dataUrl, missing: stillMissing.length ? stillMissing : null }
  }, [])

  const overlayFrom = useCallback((detected, regions) => {
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || CAPTURE_WIDTH
      canvas.height = video.videoHeight || CAPTURE_HEIGHT
    }
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!detected || !regions) return
    const scale = canvas.width / CAPTURE_WIDTH
    for (const name of REGION_KEYS) {
      const region = regions[name]
      if (!region) continue
      // Boxes come from the newest model pass, verdicts from the newest local
      // pass, so the outline tracks the card even between analyses.
      drawRegionBox(ctx, name, detected[name]?.box ?? region.box, region.positive, region.score, scale)
    }
  }, [])

  /** Single capture — the old "Capture & detect" button. */
  const capture = useCallback(async () => {
    setBusy(true)
    setFailure(null)
    say('Sending the captured frame to the detection model…')
    try {
      const frameCanvas = grabFrame()
      if (!frameCanvas) throw new Error('No camera frame is available yet.')
      const started = performance.now()
      const { detected, inference, dataUrl, missing } = await locateRegions(frameCanvas)
      if (missing) {
        const detail = describeMissing(inference, missing)
        setFailure({ ...detail, capturedImage: dataUrl })
        setResult(null)
        overlayFrom(null, null)
        say(`${detail.message} Missing: ${missing.join(', ')}.`, 'error')
        return
      }
      const analysis = analyseRegions(frameCanvas, detected, true)
      if (!analysis) throw new Error('The located wells could not be analysed.')
      setAnalysisMs(Math.round(performance.now() - started))
      setResult({ ...analysis, capturedImage: dataUrl })
      overlayFrom(detected, analysis.regions)
      liveRef.current.detected = detected
      liveRef.current.regions = analysis.regions
      say(
        `Analysis completed using ${analysis.detectionMode} at ${new Date(analysis.capturedAt).toLocaleTimeString()}.`,
        'ok',
      )
    } catch (error) {
      setResult(null)
      setFailure({ message: error.message || 'Detection failed.' })
      say(error.message || 'Detection failed.', 'error')
    } finally {
      setBusy(false)
    }
  }, [grabFrame, locateRegions, overlayFrom, say])

  /**
   * Live mode: the overlay is redrawn every presented frame, the local
   * agglutination pass re-runs a few times a second, and the model is only
   * re-asked for region boxes every LIVE_MODEL_INTERVAL_MS.
   */
  useEffect(() => {
    liveRef.current.on = live
    if (!live) {
      overlayFrom(liveRef.current.detected, liveRef.current.regions)
      return undefined
    }
    const state = liveRef.current
    const controller = new AbortController()
    let raf = 0
    let stopped = false

    const refreshBoxes = async () => {
      state.modelBusy = true
      try {
        const frameCanvas = grabFrame()
        if (frameCanvas) {
          const { detected, missing } = await locateRegions(frameCanvas, controller.signal)
          if (!stopped) {
            state.detected = missing ? null : detected
            if (missing) {
              state.regions = null
              say(`Searching for the A, B and D wells… (missing ${missing.join(', ')})`)
            }
          }
        }
      } catch (error) {
        if (!stopped && error.name !== 'AbortError') say(error.message, 'error')
      } finally {
        state.modelBusy = false
        state.lastModel = performance.now()
      }
    }

    const loop = () => {
      if (stopped) return
      const now = performance.now()
      if (!state.modelBusy && now - state.lastModel > LIVE_MODEL_INTERVAL_MS) refreshBoxes()
      if (state.detected && now >= state.nextLocal) {
        const frameCanvas = grabFrame()
        if (frameCanvas) {
          const started = performance.now()
          try {
            // One well per pass once all three are known, so a single frame of
            // analysis stays short; A, B and D refresh in about two thirds of a
            // second between them.
            const only = state.regions ? REGION_KEYS[state.cursor % REGION_KEYS.length] : null
            const analysis = analyseRegions(frameCanvas, state.detected, false, {
              only,
              previous: state.regions,
            })
            state.cursor += 1
            const elapsed = performance.now() - started
            // Never spend more than a third of the time analysing, whatever the
            // machine manages, so the preview keeps its frame rate.
            state.nextLocal = performance.now() + Math.max(LIVE_LOCAL_INTERVAL_MS, elapsed * 2)
            if (analysis) {
              state.regions = analysis.regions
              // Throttle React updates so the panel never gates the frame rate.
              if (now - state.lastPublish > 300) {
                state.lastPublish = now
                setAnalysisMs(Math.round(elapsed))
                setResult((previous) => ({
                  ...analysis,
                  annotatedImage: previous?.annotatedImage ?? null,
                  capturedImage: previous?.capturedImage ?? null,
                  live: true,
                }))
                setFailure(null)
              }
            }
          } catch {
            state.detected = null
            state.regions = null
            state.nextLocal = performance.now() + LIVE_LOCAL_INTERVAL_MS
          }
        }
      }
      overlayFrom(state.detected, state.regions)
      raf = requestAnimationFrame(loop)
    }

    state.lastModel = 0
    state.nextLocal = 0
    state.cursor = 0
    raf = requestAnimationFrame(loop)
    return () => {
      stopped = true
      controller.abort()
      cancelAnimationFrame(raf)
    }
  }, [live, grabFrame, locateRegions, overlayFrom, say])

  /** One-tap hop to the next camera on the device — the mobile-friendly path. */
  const cycleCamera = useCallback(async () => {
    if (devices.length < 2) {
      const inputs = await listCameras()
      if (inputs.length < 2) {
        say('This device only reports one camera.')
        return
      }
    }
    const list = devices.length ? devices : await listCameras()
    const current = list.findIndex((device) => device.deviceId === activeDeviceId)
    const next = list[(current + 1) % list.length]
    if (next) await selectCamera(next.deviceId)
  }, [devices, activeDeviceId, listCameras, selectCamera, say])

  const toggleLive = useCallback(() => {
    const next = !liveRef.current.on
    setLive(next)
    say(
      next
        ? 'Live detection running. The model re-locates the wells about twice a second.'
        : 'Live detection stopped.',
      'ok',
    )
  }, [say])

  const settings = track?.settings
  const resolution = settings?.width ? `${settings.width}x${settings.height}` : '—'

  return (
    <div className="shell">
      <style>{STYLES}</style>

      <header>
        <div>
          <div className="eyebrow">Clinical vision system</div>
          <h1>Blood Group Detector</h1>
        </div>
        <div className="meta">
          <div>Model: {MODEL_ID}</div>
          <div>
            Roboflow key:{' '}
            <span className={ROBOFLOW_API_KEY ? 'ok-text' : 'error'}>
              {ROBOFLOW_API_KEY ? `…${ROBOFLOW_API_KEY.slice(-4)}` : 'missing'}
            </span>
          </div>
        </div>
      </header>

      <section className="grid">
        <div className="card">
          <div className="camera" style={{ aspectRatio: aspect }}>
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              onLoadedMetadata={(event) => {
                const { videoWidth, videoHeight } = event.currentTarget
                if (videoWidth) setAspect(`${videoWidth} / ${videoHeight}`)
              }}
            />
            <canvas ref={overlayRef} className="overlay" />
            <div className="badge live">LIVE</div>
            <div className="badge stats">
              <strong>{fps}</strong> fps · {resolution}
              {analysisMs ? <span className="wide-only"> · analysis {analysisMs} ms</span> : null}
            </div>
            {track?.label ? <div className="badge name">{track.label}</div> : null}
            {track ? null : (
              <div className="camera-empty">
                {status.kind === 'error' ? status.text : 'Waiting for a camera…'}
              </div>
            )}
          </div>

          <div className="camera-picker">
            <div className="picker-row">
              <label htmlFor="camera-select">
                Camera
                <span className="count">
                  {devices.length
                    ? `${devices.length} found · ${pinned ? 'your choice, saved' : 'auto-selected'}`
                    : 'scanning…'}
                </span>
              </label>
              <select
                id="camera-select"
                value={activeDeviceId}
                disabled={!devices.length}
                onChange={(event) => selectCamera(event.target.value)}
              >
                {devices.length ? (
                  devices.map((device, index) => (
                    <option key={device.deviceId || index} value={device.deviceId}>
                      {cameraLabel(device, index)}
                    </option>
                  ))
                ) : (
                  <option value="">No camera found</option>
                )}
              </select>
            </div>
            <div className="picker-actions">
              <button type="button" className="secondary" onClick={cycleCamera} disabled={devices.length < 2}>
                Switch<span className="wide-only"> camera</span>
              </button>
              <button
                type="button"
                className={pinned ? 'secondary' : 'secondary active'}
                onClick={useAutoCamera}
                title="Forget the saved camera and pick automatically"
              >
                Auto<span className="wide-only">-select</span>
              </button>
              <button type="button" className="secondary" onClick={listCameras}>
                Rescan
              </button>
            </div>
          </div>

          <div className="controls">
            <button type="button" onClick={capture} disabled={busy}>
              {busy ? 'Analysing…' : 'Capture & detect'}
            </button>
            <button
              type="button"
              className={live ? 'toggle on' : 'toggle'}
              onClick={toggleLive}
              disabled={busy}
            >
              {live ? 'Stop live detect' : 'Live detect'}
            </button>
            <span className={status.kind === 'error' ? 'message error' : 'message'}>{status.text}</span>
          </div>
        </div>

        <aside className="card result">
          {result ? (
            <>
              <div className="eyebrow">
                Detected result · {result.detectionMode}
                {result.live ? ' · live' : ''}
              </div>
              <div className="blood">
                <span>{result.bloodGroup}</span>
              </div>
              {REGION_KEYS.map((name) => {
                const region = result.regions[name]
                return (
                  <div className="region" key={name}>
                    <div className="antigen">{name}</div>
                    <div>
                      <div className={region.positive ? 'status positive' : 'status negative'}>{region.status}</div>
                      <div className="score">
                        Agglutination score {region.score.toFixed(3)} · {region.clumps} clumps · texture{' '}
                        {region.texture}
                      </div>
                    </div>
                    <div className="score right">
                      {region.source === 'roboflow' ? `${Math.round(region.confidence * 100)}% region` : 'card layout'}
                    </div>
                  </div>
                )
              })}
              {result.annotatedImage ? (
                <>
                  <img className="result-image" src={result.annotatedImage} alt="Annotated capture" />
                  <div className="thumbs">
                    {REGION_KEYS.map((name) => {
                      const region = result.regions[name]
                      if (!region.roiImage) return null
                      return (
                        <figure key={name}>
                          <img src={region.roiImage} alt={`Anti-${name} well`} />
                          <img src={region.processedImage} alt={`Anti-${name} clump mask`} />
                          <figcaption>Anti-{name}</figcaption>
                        </figure>
                      )
                    })}
                  </div>
                  <a className="download" href={result.annotatedImage} download={`blood_${result.bloodGroup}.jpg`}>
                    Download annotated capture
                  </a>
                </>
              ) : (
                <div className="score hint">Live mode skips image encoding — stop it and capture for the report.</div>
              )}
            </>
          ) : failure ? (
            <div className="placeholder">
              <p className="error">{failure.message}</p>
              {failure.missing ? <p className="score">Missing regions: {failure.missing.join(', ')}.</p> : null}
              {failure.hint ? <p className="score">{failure.hint}</p> : null}
              {failure.capturedImage ? (
                <img className="result-image" src={failure.capturedImage} alt="Rejected capture" />
              ) : null}
            </div>
          ) : (
            <div className="placeholder">
              <p>Your analysis will appear here after capture.</p>
              <p className="score">
                Position the Anti-A, Anti-B and Anti-D wells so the card fills the frame, then capture.
              </p>
            </div>
          )}
        </aside>
      </section>
    </div>
  )
}

const STYLES = `
.shell{--bg:#08111f;--panel:#101c2d;--line:#26364d;--cyan:#36d1dc;--blue:#5b86e5;--muted:#91a2b9;
  width:100%;max-width:1180px;margin:0 auto;color:#f4f7fb;font:15px/1.5 Inter,system-ui,sans-serif;letter-spacing:0;
  text-align:left;
  padding-block:clamp(16px,4vw,34px) clamp(32px,9vw,60px);
  padding-left:max(14px,env(safe-area-inset-left));
  padding-right:max(14px,env(safe-area-inset-right))}
.shell,.shell *{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 15% 0,#19375a 0,transparent 30%),#08111f;color-scheme:dark;
  overflow-x:hidden}
#root{width:100%;max-width:none;margin:0;border:0;text-align:left;min-height:100svh;display:block}

/* ---------- header ---------- */
.shell header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;
  margin-bottom:clamp(16px,3vw,24px)}
.shell h1{margin:0;font-size:clamp(23px,6vw,42px);letter-spacing:-1.2px;font-weight:700;color:#f4f7fb;
  line-height:1.1}
.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:2px;font-weight:800;
  font-size:clamp(10px,2.4vw,12px)}
.meta,.score{color:var(--muted);font-size:12px}
.meta{text-align:right;display:grid;gap:4px;flex:0 0 auto;word-break:break-word}
.ok-text{color:#43d69d}

/* ---------- layout ---------- */
.grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.8fr);gap:22px;align-items:start}
.card{background:#101c2de8;border:1px solid var(--line);border-radius:clamp(14px,3vw,20px);
  box-shadow:0 18px 55px #0005;overflow:hidden;min-width:0}

/* ---------- camera ---------- */
.camera{position:relative;aspect-ratio:16/9;max-height:min(70svh,620px);background:#030711}
.camera video,.camera .overlay{position:absolute;inset:0;width:100%;height:100%;display:block;
  /* contain, not cover: the preview must show exactly the frame that is analysed */
  object-fit:contain}
.camera .overlay{pointer-events:none}
.camera-empty{position:absolute;inset:0;display:grid;place-items:center;padding:24px;text-align:center;
  color:var(--muted);font-size:13px}
.badge{position:absolute;padding:6px 10px;border-radius:30px;background:#09111dd9;font-size:11px;font-weight:800;
  backdrop-filter:blur(6px);max-width:calc(100% - 24px);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge.live{top:12px;left:12px}
.badge.stats{top:12px;right:12px;font-weight:600;color:var(--muted)}
.badge.stats strong{color:#f4f7fb}
.badge.name{bottom:12px;left:12px;font-weight:600;color:var(--muted);max-width:calc(100% - 24px)}
.badge.live:before{content:"";display:inline-block;width:8px;height:8px;background:#ff4d67;border-radius:50%;
  margin-right:6px;box-shadow:0 0 10px #ff4d67}

/* ---------- camera picker ---------- */
/* minmax(0,1fr) throughout: a long camera label ("camera2 0, facing back", a
   USB device name) would otherwise set the min-content width of these tracks
   and push the buttons outside the card, where overflow:hidden eats them. */
.camera-picker{padding:16px 16px 0;display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.picker-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;min-width:0}
.picker-row label{color:var(--muted);font-size:12px;font-weight:700;display:grid;gap:2px;flex:0 0 auto}
.picker-row .count{font-weight:500;font-size:11px;color:#6e819a}
select{flex:1 1 200px;width:100%;min-width:0;max-width:100%;padding:11px 12px;border:1px solid var(--line);
  border-radius:10px;color:#f4f7fb;background:#16253a;font:inherit;text-overflow:ellipsis}
.picker-actions{display:flex;gap:8px;flex-wrap:wrap;min-width:0}
.picker-actions button{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.picker-actions button.active{box-shadow:inset 0 0 0 1px #36d1dc80;color:var(--cyan)}

/* ---------- controls ---------- */
.controls{padding:16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;min-width:0}
button{border:0;border-radius:12px;padding:13px 20px;color:#fff;font:inherit;font-weight:800;cursor:pointer;
  min-height:44px;background:linear-gradient(135deg,var(--cyan),var(--blue));box-shadow:0 8px 24px #467ed44d}
button.secondary{padding:10px 13px;background:#1a2b42;box-shadow:none;font-weight:700;font-size:13px}
button.toggle{background:#1a2b42;box-shadow:none}
button.toggle.on{background:linear-gradient(135deg,#ff416c,#c9184a);box-shadow:0 8px 24px #d11d554d}
button:disabled{opacity:.45;cursor:not-allowed}
.message{color:var(--muted);flex:1 1 100%;min-width:0;font-size:13px;overflow-wrap:anywhere}
.error{color:#ff8293}

/* ---------- result ---------- */
.result{padding:clamp(18px,4vw,26px)}
.blood{display:grid;place-items:center;width:clamp(104px,26vw,130px);height:clamp(104px,26vw,130px);
  margin:18px auto 24px;border-radius:50% 50% 50% 12%;transform:rotate(45deg);
  background:linear-gradient(135deg,#ff416c,#c9184a);box-shadow:0 15px 45px #d11d554d}
.blood span{transform:rotate(-45deg);font-size:clamp(32px,8vw,40px);font-weight:900}
.placeholder{color:var(--muted);text-align:center;padding:clamp(32px,9vw,60px) 4px;display:grid;gap:12px}
.placeholder p{margin:0}
.region{display:grid;grid-template-columns:48px minmax(0,1fr) auto;gap:10px;align-items:center;padding:12px 0;
  border-top:1px solid var(--line)}
.antigen{width:42px;height:42px;display:grid;place-items:center;border-radius:12px;background:#1a2b42;
  font-weight:900}
.status{font-weight:800}
.score.right{text-align:right;white-space:nowrap}
.score.hint{margin-top:18px}
.positive{color:#43d69d}
.negative{color:#ffb454}
.result-image{width:100%;margin-top:18px;border-radius:12px;border:1px solid var(--line);display:block}
.thumbs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:14px}
.thumbs figure{margin:0;display:grid;gap:6px;min-width:0}
.thumbs img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.thumbs figcaption{color:var(--muted);font-size:11px;text-align:center;font-weight:700}
.download{display:block;margin-top:16px;text-align:center;padding:13px;border-radius:12px;background:#1a2b42;
  color:#f4f7fb;text-decoration:none;font-weight:700;font-size:13px}

/* ---------- tablet ---------- */
@media(max-width:900px){
  .grid{grid-template-columns:minmax(0,1fr)}
  .shell header{flex-direction:column;align-items:flex-start;gap:8px}
  .meta{text-align:left;grid-auto-flow:column;gap:14px}
  .camera{max-height:none}
}

/* ---------- phone ---------- */
@media(max-width:620px){
  .meta{grid-auto-flow:row;gap:2px}
  /* 16px keeps iOS Safari from zooming the page when the picker is focused */
  select{font-size:16px;flex-basis:100%}
  .picker-row label{flex:1 1 100%}
  .picker-actions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr))}
  .picker-actions button{font-size:12px;padding:10px 6px}
  .controls{padding:14px;gap:8px}
  .controls button{width:100%}
  .wide-only{display:none}
  .badge{font-size:10px;padding:5px 9px}
  .region{grid-template-columns:42px minmax(0,1fr);row-gap:4px}
  .region .score.right{grid-column:2;text-align:left}
  .antigen{width:38px;height:38px;border-radius:10px}
  .thumbs{grid-template-columns:1fr;gap:14px}
  .thumbs figure{grid-template-columns:1fr 1fr;gap:8px}
  .thumbs figcaption{grid-column:1/-1;text-align:left}
}

/* ---------- short landscape phones: keep the controls reachable ---------- */
@media(max-height:520px) and (orientation:landscape){
  .camera{max-height:62svh}
  .shell{padding-block:12px 24px}
}

@media(prefers-reduced-motion:reduce){
  .badge.live:before{box-shadow:none}
}
`
