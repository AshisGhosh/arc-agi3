"""
Video Demo Extractor for ARC-AGI-3 Human Performance Videos.

Extracts (observation, action) pairs from gameplay videos that have:
- Left side: game visuals
- Right side: terminal with frame number, action, level

Usage:
    python -m src.aria_lite.video_demo_extractor --video path/to/video.mp4 --output demos/
    python -m src.aria_lite.video_demo_extractor --folder path/to/videos/ --output demos/
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Try to import OCR library
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract")
    print("Also need tesseract binary: sudo apt install tesseract-ocr")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


@dataclass
class FrameData:
    """Extracted data from a single video frame."""
    frame_number: int
    action: int
    level: int
    game_frame: np.ndarray  # The game visual (left side)
    raw_text: str  # Raw OCR text for debugging


@dataclass
class VideoDemo:
    """Extracted demonstration from a video."""
    video_path: str
    game_id: str
    frames: list[FrameData]

    def to_demo_format(self) -> dict:
        """Convert to format compatible with DemoDataset."""
        observations = []
        actions = []
        levels = []

        for frame in self.frames:
            # Resize game frame to 64x64 (standard ARC-AGI-3 size)
            if frame.game_frame is not None:
                resized = cv2.resize(frame.game_frame, (64, 64))
                # Convert to grayscale if needed
                if len(resized.shape) == 3:
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                observations.append(resized.tolist())
            actions.append(frame.action)
            levels.append(frame.level)

        # Determine if successful (made progress)
        max_level = max(levels) if levels else 0

        return {
            "game_id": self.game_id,
            "observations": observations,
            "actions": actions,
            "rewards": [1.0 if levels[i] > levels[i-1] else 0.0
                       for i in range(1, len(levels))] + [0.0] if levels else [],
            "levels_completed": max_level,
            "won": max_level > 0,  # Consider any progress as partial success
            "total_steps": len(self.frames),
        }


class VideoExtractor:
    """Extract demonstrations from ARC-AGI-3 gameplay videos."""

    def __init__(
        self,
        ocr_engine: str = "pytesseract",
        game_region: tuple = (0, 0, 0.5, 1.0),  # Left half for game
        terminal_region: tuple = (0.5, 0, 1.0, 1.0),  # Right half for terminal
        sample_rate: int = 1,  # Extract every Nth frame
    ):
        """
        Initialize extractor.

        Args:
            ocr_engine: "pytesseract" or "easyocr"
            game_region: (x1, y1, x2, y2) as fractions of frame size
            terminal_region: (x1, y1, x2, y2) as fractions of frame size
            sample_rate: Extract every Nth frame (1 = all frames)
        """
        self.ocr_engine = ocr_engine
        self.game_region = game_region
        self.terminal_region = terminal_region
        self.sample_rate = sample_rate

        if ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'], gpu=True)
        else:
            self.reader = None

    def extract_regions(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract game and terminal regions from frame."""
        h, w = frame.shape[:2]

        # Game region (left side)
        gx1 = int(w * self.game_region[0])
        gy1 = int(h * self.game_region[1])
        gx2 = int(w * self.game_region[2])
        gy2 = int(h * self.game_region[3])
        game_frame = frame[gy1:gy2, gx1:gx2]

        # Terminal region (right side)
        tx1 = int(w * self.terminal_region[0])
        ty1 = int(h * self.terminal_region[1])
        tx2 = int(w * self.terminal_region[2])
        ty2 = int(h * self.terminal_region[3])
        terminal_frame = frame[ty1:ty2, tx1:tx2]

        return game_frame, terminal_frame

    def ocr_terminal(self, terminal_frame: np.ndarray) -> str:
        """Run OCR on terminal region."""
        # Preprocess for better OCR
        gray = cv2.cvtColor(terminal_frame, cv2.COLOR_BGR2GRAY) if len(terminal_frame.shape) == 3 else terminal_frame

        # Threshold to make text clearer
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Invert if terminal has light text on dark background
        if np.mean(thresh) < 127:
            thresh = 255 - thresh

        if self.ocr_engine == "easyocr" and self.reader is not None:
            results = self.reader.readtext(thresh)
            text = " ".join([r[1] for r in results])
        elif OCR_AVAILABLE:
            # Use pytesseract
            text = pytesseract.image_to_string(thresh, config='--psm 6')
        else:
            text = ""

        return text

    def parse_terminal_text(self, text: str) -> tuple[int, int, int]:
        """
        Parse frame number, action, and level from terminal text.

        Expected format variations:
        - "Frame: 123 Action: 4 Level: 2"
        - "frame=123 action=4 level=2"
        - "123 4 2"

        Returns:
            (frame_number, action, level) or (-1, -1, -1) if parsing fails
        """
        text = text.lower()

        # Try different patterns
        patterns = [
            # Pattern 1: "frame: X action: Y level: Z"
            r'frame[:\s]+(\d+).*action[:\s]+(\d+).*level[:\s]+(\d+)',
            # Pattern 2: "frame=X action=Y level=Z"
            r'frame\s*=\s*(\d+).*action\s*=\s*(\d+).*level\s*=\s*(\d+)',
            # Pattern 3: Just numbers on separate lines
            r'(\d+)\s+(\d+)\s+(\d+)',
            # Pattern 4: Action and level only
            r'action[:\s]+(\d+).*level[:\s]+(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    return int(groups[0]), int(groups[1]), int(groups[2])
                elif len(groups) == 2:
                    return -1, int(groups[0]), int(groups[1])

        # Try to find any numbers
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 3:
            return int(numbers[0]), int(numbers[1]), int(numbers[2])
        elif len(numbers) >= 2:
            return -1, int(numbers[0]), int(numbers[1])

        return -1, -1, -1

    def extract_from_video(
        self,
        video_path: str,
        game_id: Optional[str] = None,
    ) -> VideoDemo:
        """
        Extract demonstration from a video file.

        Args:
            video_path: Path to video file
            game_id: Game identifier (inferred from filename if not provided)

        Returns:
            VideoDemo with extracted frames
        """
        video_path = Path(video_path)

        # Infer game_id from filename if not provided
        if game_id is None:
            # Try to find game ID pattern (e.g., ls20, vc33, ft09)
            match = re.search(r'(ls\d+|vc\d+|ft\d+)', video_path.stem, re.IGNORECASE)
            game_id = match.group(1).lower() if match else "unknown"

        print(f"Extracting from {video_path} (game: {game_id})")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        frame_idx = 0
        prev_action = -1

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {total_frames} frames, {fps:.1f} FPS")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                continue

            # Extract regions
            game_frame, terminal_frame = self.extract_regions(frame)

            # OCR terminal
            text = self.ocr_terminal(terminal_frame)

            # Parse data
            frame_num, action, level = self.parse_terminal_text(text)

            # Only keep frames where action changed or is valid
            if action >= 0 and (action != prev_action or len(frames) == 0):
                frame_data = FrameData(
                    frame_number=frame_num if frame_num >= 0 else frame_idx,
                    action=action,
                    level=level if level >= 0 else 0,
                    game_frame=game_frame,
                    raw_text=text,
                )
                frames.append(frame_data)
                prev_action = action

                if len(frames) % 50 == 0:
                    print(f"  Extracted {len(frames)} action frames...")

            frame_idx += 1

        cap.release()
        print(f"  Total: {len(frames)} action frames extracted")

        return VideoDemo(
            video_path=str(video_path),
            game_id=game_id,
            frames=frames,
        )

    def extract_from_folder(
        self,
        folder_path: str,
        extensions: list[str] = [".mp4", ".avi", ".mov", ".mkv"],
    ) -> list[VideoDemo]:
        """Extract demos from all videos in a folder."""
        folder = Path(folder_path)
        demos = []

        video_files = []
        for ext in extensions:
            video_files.extend(folder.glob(f"*{ext}"))
            video_files.extend(folder.glob(f"*{ext.upper()}"))

        print(f"Found {len(video_files)} video files")

        for video_path in sorted(video_files):
            try:
                demo = self.extract_from_video(str(video_path))
                if len(demo.frames) > 0:
                    demos.append(demo)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        return demos


def save_demos(demos: list[VideoDemo], output_dir: str) -> None:
    """Save extracted demos to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by game
    by_game: dict[str, list] = {}
    for demo in demos:
        game_id = demo.game_id
        if game_id not in by_game:
            by_game[game_id] = []
        by_game[game_id].append(demo.to_demo_format())

    # Save each game's demos
    for game_id, game_demos in by_game.items():
        output_file = output_path / f"{game_id}_human_demos.json"

        data = {
            "game_id": game_id,
            "num_demos": len(game_demos),
            "demos": game_demos,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        successful = sum(1 for d in game_demos if d["levels_completed"] > 0)
        print(f"Saved {len(game_demos)} demos ({successful} successful) to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract demos from ARC-AGI-3 gameplay videos")
    parser.add_argument("--video", "-v", type=str, help="Single video file to process")
    parser.add_argument("--folder", "-f", type=str, help="Folder of videos to process")
    parser.add_argument("--output", "-o", type=str, default="demos/human", help="Output directory")
    parser.add_argument("--game-id", "-g", type=str, help="Override game ID")
    parser.add_argument("--sample-rate", "-s", type=int, default=5, help="Sample every Nth frame")
    parser.add_argument("--ocr", type=str, default="pytesseract", choices=["pytesseract", "easyocr"])
    parser.add_argument("--preview", "-p", action="store_true", help="Show preview of extraction")

    args = parser.parse_args()

    if not args.video and not args.folder:
        print("Error: Must provide --video or --folder")
        return

    if not OCR_AVAILABLE and not EASYOCR_AVAILABLE:
        print("Error: No OCR library available!")
        print("Install pytesseract: pip install pytesseract")
        print("  Also need: sudo apt install tesseract-ocr")
        print("Or install easyocr: pip install easyocr")
        return

    extractor = VideoExtractor(
        ocr_engine=args.ocr,
        sample_rate=args.sample_rate,
    )

    demos = []

    if args.video:
        demo = extractor.extract_from_video(args.video, game_id=args.game_id)
        if len(demo.frames) > 0:
            demos.append(demo)

            if args.preview and demo.frames:
                # Show first frame
                frame = demo.frames[0]
                print(f"\nFirst frame preview:")
                print(f"  Frame: {frame.frame_number}")
                print(f"  Action: {frame.action}")
                print(f"  Level: {frame.level}")
                print(f"  Raw text: {frame.raw_text[:100]}...")

    if args.folder:
        demos = extractor.extract_from_folder(args.folder)

    if demos:
        save_demos(demos, args.output)

        # Summary
        total_frames = sum(len(d.frames) for d in demos)
        successful = sum(1 for d in demos if any(f.level > 0 for f in d.frames))
        print(f"\nExtraction complete:")
        print(f"  Videos processed: {len(demos)}")
        print(f"  Total action frames: {total_frames}")
        print(f"  Demos with progress: {successful}")
    else:
        print("No demos extracted!")


if __name__ == "__main__":
    main()
