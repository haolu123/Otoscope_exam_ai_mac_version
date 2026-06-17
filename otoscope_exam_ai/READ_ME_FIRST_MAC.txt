READ ME FIRST - macOS unsigned app
==================================

This package was built without Apple Developer signing and notarization.
macOS may show a security warning the first time you open the app.

Package layout:

  Otoscope Exam AI.app
  ai_worker/
  videos/
  result/
  ai_output/

Keep these items together in the same folder. Do not move only the .app by
itself, because AI analysis needs the ai_worker folder and videos folder.

If the videos folder only contains PUT_VIDEOS_HERE.txt, copy the real videos
folder into this package before using the app. The videos folder must contain
the seven category folders:

  AOM
  Effusion
  Normal
  Perforation
  Retraction
  Tubes
  Tympanosclerosis

How to open the app
-------------------

Option 1: Right-click Open

1. Unzip the package.
2. Right-click "Otoscope Exam AI.app".
3. Click "Open".
4. If macOS shows a warning, click "Open" again.

Option 2: Remove quarantine in Terminal

1. Open Terminal.
2. Run this command, replacing the path with the folder you unzipped:

   xattr -dr com.apple.quarantine "/path/to/otoscope_exam_ai_mac"

3. Double-click "Otoscope Exam AI.app".

If macOS says the app is damaged
--------------------------------

This usually means macOS quarantine blocked an unsigned app. Run:

   xattr -dr com.apple.quarantine "/path/to/otoscope_exam_ai_mac"

Then try opening the app again.

Privacy note
------------

Results are saved locally in the result folder. AI output images are cached
locally in the ai_output folder.
