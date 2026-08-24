# Cricket Shot Quality AI - Folder Structure Explanation

## 📁 Main Data Folder Structure

Humara project **10 shot classes** ka data organize karta hai:
- cover, defense, flick, hook, late_cut, lofted, pull, square_cut, straight, sweep

**Har shot class folder mein kya hai:**

---

## 🎬 Video Files (Directly in folder)

### Original Dataset Videos
```
video1.mp4
video2.mp4
video3.mp4
video4.mp4
video5.mp4
```
**Kya hai ye?** 
- Real cricket shot videos from dataset
- Har shot class ke 5 examples
- Original videos jo HuggingFace se download kiye

**Size:** ~5MB each
**Purpose:** Model training aur testing ke liye

---

### Skeleton Overlay Videos
```
skeleton_video1_overlay.mp4
skeleton_video2_overlay.mp4
skeleton_video3_overlay.mp4
skeleton_video4_overlay.mp4
skeleton_video5_overlay.mp4
```
**Kya hai ye?**
- Original video + skeleton visualization combined
- Green lines se body joints connected dikhte hain
- Orange/Cyan dots se key body points dikhte hain
- **IMPORTANT:** Sirf batsman pe skeleton, umpire/fielders pe nahi

**Size:** ~5MB each
**Purpose:** Visualization - dekhne ke liye ki skeleton detection kaise kaam kar raha hai

---

## 📂 Pipeline Folders (Processing Steps)

Har video ka ek `videoX_pipeline/` folder hai jo processing ke har step ko save karta hai:

### Example: `video1_pipeline/`

---

### 📄 00_metadata.json
**Kya hai ye?**
Video ki basic information JSON format mein:
```json
{
  "video_name": "video1.mp4",
  "shot_class": "cover",
  "original_fps": 25,
  "resolution": "1280x720",
  "total_frames_in_video": 50,
  "extracted_frames_count": 30,
  "frames_used_for_prediction": 30,
  "skeleton_landmarks": 13
}
```

**Kaam kya hai?**
- Video ka technical data track karna
- Model ko kitne frames mile ye record karna
- Resolution aur FPS information store karna

---

### 📁 01_extracted_frames/
**Kya hai ye?**
```
frame_000.jpg
frame_001.jpg
frame_002.jpg
...
frame_029.jpg
```
Total: **30 images**

**Kaam kya hai?**
- Video ke **middle 60%** se 30 frames nikale gaye
- Kyunki shot ki important action middle mein hoti hai
- Ye exact 30 frames **model ko input** jaate hain prediction ke liye

**Detail:**
- Original video mein 50-150 frames ho sakte hain
- Hum 20%-80% range se evenly 30 frames select karte hain
- Start aur end skip karte hain kyunki wahan kuch action nahi hota

**Size:** ~100KB per frame

---

### 📄 02_skeleton_keypoints.json
**Kya hai ye?**
Har frame mein detect hue body landmarks ka data:
```json
{
  "total_frames": 30,
  "detected_frames_count": 9,
  "detection_rate": "30.0%",
  "detected_frame_indices": [0, 4, 9, 10, 11, 12, 13, 14, 15],
  "keypoints_data": [
    {
      "frame_index": 0,
      "landmarks": {
        "0": {"x": 0.484, "y": 0.663, "z": -0.23, "visibility": 0.98},
        "11": {"x": 0.421, "y": 0.542, "z": -0.15, "visibility": 0.95},
        ...
      }
    }
  ]
}
```

**Kaam kya hai?**
- **13 body landmarks** ki exact positions save karna
- x, y, z coordinates (normalized 0-1 range)
- visibility = confidence score (kitne clearly dikha)
- Kon se frames mein batsman properly detect hua

**13 Landmarks kaun se hain?**
1. Nose (naak)
2. Left Shoulder (baaya kandha)
3. Right Shoulder (daaya kandha)
4. Left Elbow (baaya kohni)
5. Right Elbow (daaya kohni)
6. Left Wrist (baaya kalai)
7. Right Wrist (daaya kalai)
8. Left Hip (baaya kamar)
9. Right Hip (daaya kamar)
10. Left Knee (baaya ghutna)
11. Right Knee (daaya ghutna)
12. Left Ankle (baaya takna)
13. Right Ankle (daaya takna)

**Size:** ~20-30KB

---

### 📁 03_skeleton_overlay_frames/
**Kya hai ye?**
```
skeleton_frame_000.jpg
skeleton_frame_001.jpg
...
skeleton_frame_029.jpg
```
Total: **30 images**

**Kaam kya hai?**
- Har frame pe skeleton drawn hai
- Original frame + skeleton visualization
- Green lines = body joint connections
- Colored dots = landmark positions
- Text shows: "Frame X/30 - DETECTED" ya "NO_DETECTION"

**Difference from 01_extracted_frames:**
- `01_extracted_frames/` = Clean frames (sirf original)
- `03_skeleton_overlay_frames/` = Frames with skeleton drawn

**Purpose:** 
- Visual verification ke liye
- Dekh sakte hain ki detection sahi hai ya nahi
- Debugging ke liye helpful

**Size:** ~120KB per frame

---

### 📁 04_comparison_frames/
**Kya hai ye?**
```
comparison_frame_000.jpg
comparison_frame_004.jpg
comparison_frame_009.jpg
...
```
Total: **9-25 images** (sirf detected frames)

**Kaam kya hai?**
- Side-by-side comparison
- Left side = Original frame
- Right side = Skeleton overlay
- **Sirf un frames ke jo detect hue** (jahan batsman clearly dikha)

**Purpose:**
- Quality check ke liye
- Sir ko dikhane ke liye ki skeleton kitna accurate hai
- Paper/presentation mein use kar sakte hain

**Size:** ~200KB per frame (double width)

---

### 📄 PIPELINE_SUMMARY.txt
**Kya hai ye?**
Complete processing ka human-readable summary:
```
PIPELINE SUMMARY - COVER - VIDEO 1
============================================================

INPUT VIDEO
-----------
File: video1.mp4
Resolution: 1280x720
FPS: 25
Total Frames in Video: 50

STEP 1: FRAME EXTRACTION
-------------------------
Frames Extracted: 30

STEP 2: POSE DETECTION
-----------------------
Frames with Batsman Detected: 9/30
Detection Rate: 30.0%

FRAMES USED FOR PREDICTION
---------------------------
Frame Indices Used: [0, 4, 9, 10, 11, 12, 13, 14, 15]
```

**Kaam kya hai?**
- Poori processing ka summary ek jagah
- Easily readable format
- Sir ko explain karne ke liye perfect

**Size:** ~2KB

---

## 🎯 Complete Folder Structure Example

```
data/
├── cover/
│   ├── video1.mp4                          ← Original dataset video
│   ├── video2.mp4
│   ├── ...
│   ├── skeleton_video1_overlay.mp4         ← Skeleton visualization video
│   ├── skeleton_video2_overlay.mp4
│   ├── ...
│   ├── video1_pipeline/                    ← Processing steps
│   │   ├── 00_metadata.json                ← Video info
│   │   ├── 01_extracted_frames/            ← 30 original frames
│   │   │   ├── frame_000.jpg
│   │   │   ├── frame_001.jpg
│   │   │   └── ...
│   │   ├── 02_skeleton_keypoints.json      ← Landmark coordinates
│   │   ├── 03_skeleton_overlay_frames/     ← 30 frames with skeleton
│   │   │   ├── skeleton_frame_000.jpg
│   │   │   ├── skeleton_frame_001.jpg
│   │   │   └── ...
│   │   ├── 04_comparison_frames/           ← Side-by-side comparisons
│   │   │   ├── comparison_frame_000.jpg
│   │   │   ├── comparison_frame_004.jpg
│   │   │   └── ...
│   │   └── PIPELINE_SUMMARY.txt            ← Complete summary
│   ├── video2_pipeline/
│   ├── video3_pipeline/
│   ├── video4_pipeline/
│   └── video5_pipeline/
├── defense/
├── flick/
├── hook/
├── late_cut/
├── lofted/
├── pull/
├── square_cut/
├── straight/
└── sweep/
```

---

## 📊 Summary Numbers

### Per Shot Class:
- **10 video files** (5 original + 5 skeleton overlay)
- **5 pipeline folders** (ek per video)
- **Per pipeline folder:**
  - 1 metadata JSON
  - 30 extracted frame images
  - 1 skeleton keypoints JSON
  - 30 skeleton overlay images
  - 9-25 comparison images (varies)
  - 1 summary text file

### Total Project:
- **10 shot classes**
- **50 original videos**
- **50 skeleton overlay videos**
- **50 pipeline folders**
- **~9000+ individual frame images** across all pipelines
- **Complete traceability** of every processing step

---

## 🎓 Sir Ko Kya Batana Hai

### 1. Data Organization
"Sir, humne complete dataset ko organize kiya hai. Har shot class (cover, defense, etc.) ke liye 5 videos hain. Total 50 cricket shot videos."

### 2. Skeleton Detection
"Humne advanced skeleton detection implement kiya hai jo **sirf batsman ko track** karta hai, umpire ya fielders ko nahi. Har video ka skeleton overlay version bhi banaya hai dekhne ke liye."

### 3. Processing Pipeline
"Har video ki complete processing pipeline save hai. Har step ka output alag folder mein organized hai:
- Kaun se frames model ko input gaye
- Kahan kahan batsman detect hua
- Skeleton landmarks ka exact position data
- Visual comparison images"

### 4. Traceability
"Koi bhi confusion ho toh hum easily check kar sakte hain ki kis frame se kya prediction hua, kyunki har intermediate step saved hai."

### 5. Ready for Model Training
"Ye 30 frames per video model ko input jayenge. Humne already extraction kar ke save kar diya hai, so training fast hogi."

---

## 💡 Key Advantages

1. **Transparency:** Har processing step visible hai
2. **Debugging:** Agar kuch galat ho toh easily identify kar sakte hain
3. **Reproducibility:** Exact same steps repeat kar sakte hain
4. **Documentation:** Automatically generated summaries
5. **Visual Verification:** Comparison images se quality check easy hai
6. **Model Ready:** Processed frames directly model ko feed kar sakte hain

---

## ✅ Quality Features

### Batsman-Only Detection
- ✅ Multi-person detection algorithm
- ✅ Size-based filtering (batsman bada hota hai)
- ✅ Position-based filtering (batsman niche frame mein)
- ✅ Temporal consistency (same person track karna)
- ✅ Confidence-based filtering

### Data Quality
- ✅ 30 frames per video (optimal for model)
- ✅ Middle 60% extraction (important action capture)
- ✅ 13 key body landmarks (cricket-specific)
- ✅ Normalized coordinates (0-1 range)
- ✅ Visibility scores included

---

**Created by:** Vipul's BTP Project
**Purpose:** Cricket Shot Quality Assessment using AI
**Date:** July 2026
