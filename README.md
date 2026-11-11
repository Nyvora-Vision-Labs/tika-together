# Tika Together

[Open Tika Together](https://nyvora-vision-labs.github.io/tika-together/) · [GitHub repository](https://github.com/Nyvora-Vision-Labs/tika-together)

A responsive, buildless Bhai Tika photo upload prototype. Open `dist/index.html` in a browser, or serve `dist` with `python3 -m http.server 8000 --directory dist`.

Supports separate sister and brother photos, drag and drop, previews, replacement, removal, image decoding validation, and a 10 MB limit per photo. Accepted formats: JPEG, PNG, WebP. Photos remain in browser memory and are cleared on reload; there is no upload server or persistent storage.

Tika placement runs in the browser (`dist/tika.js`, algorithm documented at the top of the file). It loads MediaPipe Face Landmarker from jsDelivr and the model from Google's storage the first time, then finds faces, measures each forehead from the face mesh, and paints a seven-colour saptarangi tika. There are two modes: **separate photos** (one face per photo, combined into a side-by-side keepsake) and **together** (one photo, a tika on every main face, at least two required). Each person also gets a marigold + makhamali mala placed from their shoulders (MediaPipe Pose and body segmentation). After creating, **Adjust mala** opens an editor: drag the two handles to set where the mala starts and ends (it keeps a fixed length, like a real garland), drag the flowers to move it, and make it bigger or smaller (40–250 %) with the size slider, the −/+ buttons, the +/− keys, or a pinch (touch or trackpad). Resizing scales the whole mala from the neck, so it keeps its shape. The output is a PNG you can download. Photos never leave the device.

The hero uses illustrative SVG artwork, not generated photo results. Fonts use Google Fonts with local fallbacks.
