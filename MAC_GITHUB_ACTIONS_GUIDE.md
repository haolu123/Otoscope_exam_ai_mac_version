# Build the macOS App with GitHub Actions

This repository can build a macOS `.app` without owning a Mac by using a GitHub
Actions macOS runner.

## Steps

1. Create a new GitHub repository under `https://github.com/haolu123`.
2. Install Git LFS locally if it is not already installed.
3. Push this project to that repository.
4. Open the repository on GitHub.
5. Go to the **Actions** tab.
6. Run **Build macOS Otoscope Exam AI**.
7. Download the artifact named `otoscope_exam_ai_mac`.

Example first push:

```powershell
cd C:\Users\haolu\Desktop\otoscope_exam
git init
git lfs install
git add .gitattributes .gitignore .github MAC_GITHUB_ACTIONS_GUIDE.md otoscope_exam_ai
git commit -m "Add macOS GitHub Actions build"
git branch -M main
git remote add origin https://github.com/haolu123/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

Replace `YOUR_REPOSITORY_NAME` with the repository you created.

The artifact contains:

```text
otoscope_exam_ai_mac/
├─ READ_ME_FIRST_MAC.txt
├─ Otoscope Exam AI.app
├─ ai_worker/
├─ videos/
├─ result/
└─ ai_output/
```

The GitHub Actions build does not need to upload the video dataset. If the
repository does not include `otoscope_exam_ai/videos/`, the artifact will contain
an empty `videos/` folder with `PUT_VIDEOS_HERE.txt`.

Before giving the app to a user, copy the full seven-category `videos/` folder
into the same folder as `Otoscope Exam AI.app`, or zip the app package and videos
together locally.

## Unsigned App Notice

This build is not Apple-signed or notarized. macOS users may need to right-click
the app and choose **Open**, or remove quarantine with:

```bash
xattr -dr com.apple.quarantine "/path/to/otoscope_exam_ai_mac"
```

The same instructions are included in `READ_ME_FIRST_MAC.txt` inside the zip.

## Large Files

The model checkpoints are large and use Git LFS. The video dataset does not need
to be uploaded to GitHub.
