import os
import cv2
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

REPO_DIR = "/home/maniskill/Multi-viewpoint_Generalization"
OUTPUT_PPTX = os.path.join(REPO_DIR, "btp_submission", "BTP_Final_Defense_Presentation.pptx")
ROOT_PPTX = os.path.join(REPO_DIR, "BTP_Final_Defense_Presentation.pptx")
ARTIFACT_PPTX = "/root/.gemini/antigravity-cli/brain/4509f9ab-8307-466a-8346-a6d6f96fae85/BTP_Final_Defense_Presentation.pptx"

def resolve_img(basename):
    candidates = [
        os.path.join(REPO_DIR, "btp_submission", "latex_report", "figures", basename),
        os.path.join(REPO_DIR, "assets", "figures", basename),
        os.path.join(REPO_DIR, "reports", basename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

def extract_poster(video_path, poster_name):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    poster_path = f"/tmp/{poster_name}"
    if ret:
        cv2.imwrite(poster_path, frame)
        return poster_path
    return None

# Palette
C_NAVY = RGBColor(15, 23, 42)
C_DARK_BLUE = RGBColor(30, 41, 59)
C_PRIMARY = RGBColor(37, 99, 235)
C_ACCENT = RGBColor(14, 165, 233)
C_SUCCESS = RGBColor(16, 185, 129)
C_WARNING = RGBColor(245, 158, 11)
C_DANGER = RGBColor(239, 68, 68)
C_LIGHT_BG = RGBColor(248, 250, 252)
C_CARD_BG = RGBColor(241, 245, 249)
C_WHITE = RGBColor(255, 255, 255)
C_MUTED = RGBColor(100, 116, 139)
C_BORDER = RGBColor(203, 213, 225)

def style_cell(cell, text, font_size=10, bold=False, color=C_NAVY, align=PP_ALIGN.CENTER, bg_color=None):
    cell.text = text
    if bg_color:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.color.rgb = color

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, category: str, title: str):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_PRIMARY

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(23)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY

    def add_card(slide, left, top, width, height, title, items, badge=None, badge_color=C_PRIMARY):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD_BG
        card.line.color.rgb = C_BORDER
        card.line.width = Pt(1)

        tbox = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), height - Inches(0.36))
        tf = tbox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = C_NAVY

        for lead, body in items:
            p2 = tf.add_paragraph()
            p2.space_before = Pt(6)
            p2.font.size = Pt(11)
            
            run_lead = p2.add_run()
            run_lead.text = f"• {lead}: "
            run_lead.font.bold = True
            run_lead.font.color.rgb = C_NAVY
            
            run_body = p2.add_run()
            run_body.text = body
            run_body.font.color.rgb = C_DARK_BLUE

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Hero Theme) [~1 min]
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.color.rgb = C_NAVY

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "BACHELOR'S THESIS PROJECT FINAL DEFENSE"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_ACCENT

    p2 = tf1.add_paragraph()
    p2.text = "Continuous Multi-Viewpoint Generalization in Deep Visual Robotic Manipulation"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = C_WHITE
    p2.space_before = Pt(10)

    p3 = tf1.add_paragraph()
    p3.text = "Overcoming the 2D Spatial Collapse via 3D Metric Geometry and Kinematic Decoupling"
    p3.font.size = Pt(17)
    p3.font.color.rgb = RGBColor(203, 213, 225)
    p3.space_before = Pt(8)

    meta_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.8), Inches(11.333), Inches(1.8))
    meta_card.fill.solid()
    meta_card.fill.fore_color.rgb = C_DARK_BLUE
    meta_card.line.color.rgb = C_PRIMARY

    tb_meta = s1.shapes.add_textbox(Inches(1.3), Inches(5.0), Inches(10.7), Inches(1.4))
    tf_m = tb_meta.text_frame
    tf_m.word_wrap = True

    p_c = tf_m.paragraphs[0]
    p_c.text = "Candidate: Durgesh Balkrishna Shelke  |  Supervisor: Prof. Shivashankar B. Nair"
    p_c.font.size = Pt(14)
    p_c.font.bold = True
    p_c.font.color.rgb = C_WHITE

    p_d = tf_m.add_paragraph()
    p_d.text = "Department of Computer Science and Engineering, Indian Institute of Technology Guwahati"
    p_d.font.size = Pt(12)
    p_d.font.color.rgb = RGBColor(148, 163, 184)
    p_d.space_before = Pt(4)

    p_t = tf_m.add_paragraph()
    p_t.text = "Format: 15–20 Minute Thesis Examination  |  Academic Year: 2026"
    p_t.font.size = Pt(11)
    p_t.font.bold = True
    p_t.font.color.rgb = C_ACCENT
    p_t.space_before = Pt(4)

    s1.notes_slide.notes_text_frame.text = (
        "[0:00 - 1:00] Good morning, respected committee members and supervisor Prof. Shivashankar B. Nair. "
        "Today I present my Bachelor's Thesis on Continuous Multi-Viewpoint Generalization in Deep Visual Robotic Manipulation. "
        "In this work, we uncover a fundamental failure mode in standard 2D visuomotor policies—the Continuous 2D Collapse—and "
        "present a mathematically grounded 3D geometric solution that achieves continuous viewpoint invariance."
    )

    # =========================================================================
    # SLIDE 2: The Core Problem: Continuous 2D Collapse [~1.5 min]
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Research Motivation & Core Paradox", "The Continuous 2D Collapse in Deep Visuomotor Policies")

    add_card(
        s2, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.3),
        "The Illusion of Multi-Camera Generalization",
        [
            ("The 2D Paradigm", "Prior state-of-the-art (e.g. OmniView, NeurIPS'24) trains policies across multiple discrete camera views (0°, 45°, 90°, 135°), claiming multi-viewpoint generalization."),
            ("The Performance Cliff", "While scoring 90.7% on trained anchor angles, shifting the camera by just 22.5° drops success to 4.3%—an 86.4% collapse gap."),
            ("Extrinsics Conditioning Fails", "Modulating 2D CNN features with camera extrinsics via FiLM layers only recovers 24.4% success, leaving an unbridgeable 58.7% gap."),
            ("Key Paradox", "2D representations interpolate visual style smoothly, but collapse catastrophically on physical 3D spatial actions.")
        ]
    )

    img_v = resolve_img("camera_rendered_views.png")
    if os.path.exists(img_v):
        s2.shapes.add_picture(img_v, Inches(7.9), Inches(1.6), width=Inches(4.6))

    # Stat callout cards below image
    c_stat1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.9), Inches(4.7), Inches(2.2), Inches(2.2))
    c_stat1.fill.solid()
    c_stat1.fill.fore_color.rgb = RGBColor(254, 242, 242)
    c_stat1.line.color.rgb = C_DANGER
    tf_s1 = c_stat1.text_frame
    tf_s1.word_wrap = True
    p = tf_s1.paragraphs[0]
    p.text = "86.4%"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = C_DANGER
    p.alignment = PP_ALIGN.CENTER
    p2 = tf_s1.add_paragraph()
    p2.text = "OmniView (2D)\nGeneralization Gap"
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = C_NAVY
    p2.alignment = PP_ALIGN.CENTER

    c_stat2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.3), Inches(4.7), Inches(2.2), Inches(2.2))
    c_stat2.fill.solid()
    c_stat2.fill.fore_color.rgb = RGBColor(254, 243, 199)
    c_stat2.line.color.rgb = C_WARNING
    tf_s2 = c_stat2.text_frame
    tf_s2.word_wrap = True
    p = tf_s2.paragraphs[0]
    p.text = "58.7%"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = C_WARNING
    p.alignment = PP_ALIGN.CENTER
    p2 = tf_s2.add_paragraph()
    p2.text = "Extrinsics FiLM (2D)\nGeneralization Gap"
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = C_NAVY
    p2.alignment = PP_ALIGN.CENTER

    s2.notes_slide.notes_text_frame.text = (
        "[1:00 - 2:30] We begin with the core paradox. In computer vision, multi-view distillation is assumed to produce viewpoint-invariant policies. "
        "We replicated OmniView from NeurIPS 2024 on ManiSkill 3.0. On the 4 trained viewpoints, the policy achieves 90.7% success. "
        "However, when we shift the camera by just 22.5 degrees between anchors, performance collapses to 4.3%. "
        "Even when conditioning 2D convolutions on camera pose with FiLM layers, the gap remains at 58.7%. We call this the Continuous 2D Collapse."
    )

    # =========================================================================
    # SLIDE 3: Theoretical Root Cause [~1.5 min]
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Theoretical Root Cause", "Why 2D Policies Fail: Projective Parallax vs 3D Metric Space")

    img_geom = resolve_img("camera_geometry_diagram.png")
    if os.path.exists(img_geom):
        s3.shapes.add_picture(img_geom, Inches(0.8), Inches(1.6), width=Inches(5.0))

    add_card(
        s3, Inches(6.1), Inches(1.6), Inches(6.4), Inches(5.3),
        "The Geometric Origin of Failure",
        [
            ("Appearance vs Action Space", "2D vision models learn many-to-one invariant representations. In contrast, robotics requires precise, millimeter-accurate 3D physical actions."),
            ("Projective Parallax", "Rotating the camera shifts object pixel positions by ~80 pixels. To the 2D CNN, this looks like the cube moved, causing the arm to grasp empty space."),
            ("Stationary Physical Reality", "In physical Cartesian space, the cube coordinates never moved (Δx_world = 0). The shift is purely a projective artifact."),
            ("The Metric Solution", "Transforming RGB-D points into the robot's base frame: (X_b, Y_b, Z_b) = T_cam^base · K^-1 · [u, v, 1]^T · d guarantees true physical invariance.")
        ]
    )

    s3.notes_slide.notes_text_frame.text = (
        "[2:30 - 4:00] Why does this collapse happen? As illustrated in the ray diagram on the left, 2D convolutions operate on pixel grids. "
        "When the camera rotates, the object's pixel projection moves by up to 80 pixels, even though its physical position in world coordinates is unchanged. "
        "Standard data augmentations cannot resolve this projective parallax. "
        "The solution is purely geometric: by unprojecting depth pixels into the robot's base frame using camera extrinsics, the representation becomes invariant to viewpoint."
    )

    # =========================================================================
    # SLIDE 4: Proposed Solution: PointNet++ Visuomotor Policy [~1.5 min]
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Proposed Methodology", "PointNet++ Visuomotor Policy with Base-Frame Metric Invariance")

    # 3 Architectural Columns
    add_card(
        s4, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "1. Base Metric Lifting",
        [
            ("Unprojection", "Lifts RGB-D depth frames directly into robot base coordinates (X_b, Y_b, Z_b)."),
            ("FPS Downsampling", "Farthest Point Sampling extracts N=1200 points covering scene and robot."),
            ("Kinematic Decoupling", "Separates sensor viewpoint from manipulator base coordinates.")
        ]
    )

    add_card(
        s4, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "2. Hierarchical PointNet++",
        [
            ("Set Abstraction", "3 hierarchical grouping stages at radii r = 0.05 m, 0.10 m, 0.20 m."),
            ("Feature Extraction", "Shared MLPs with channels [64, 128, 256] capture local surface geometry."),
            ("Global Latent", "Symmetric max-pooling yields a robust 128-D spatial feature vector z_geo.")
        ]
    )

    add_card(
        s4, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "3. Decoupled Action Head",
        [
            ("Proprioception Fusion", "Concatenates 128-D geometry vector with joint angles q and velocities q_dot."),
            ("Actor MLP", "Two hidden layers (256 units) predict end-effector Cartesian velocities (Δx, Δy, Δz, Δyaw)."),
            ("PPO Training", "Trained end-to-end via PPO with 4096 rollout buffer and minibatch 128.")
        ]
    )

    s4.notes_slide.notes_text_frame.text = (
        "[4:00 - 5:30] Here is our proposed architecture. First, raw depth frames are unprojected into the robot base frame and sampled down to 1200 points. "
        "Second, a hierarchical PointNet++ encoder processes the point cloud through three Set Abstraction layers, extracting a compact 128-dimensional spatial embedding. "
        "Finally, an Actor MLP combines this geometry vector with joint proprioception to predict Cartesian end-effector velocities. "
        "This completely decouples camera perspective from robot action execution."
    )

    # =========================================================================
    # SLIDE 5: Benchmark Breakthrough: PickCube-v1 [~2 min]
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Experimental Breakthrough", "PickCube-v1 Benchmark: 86.4% Collapse Slashed to 1.3%")

    img_b = resolve_img("phase10_interpolated_viewpoints_benchmark.png")
    if os.path.exists(img_b):
        s5.shapes.add_picture(img_b, Inches(0.8), Inches(1.6), width=Inches(6.0))

    # Right: Comparison Table & Highlights
    table_bench = s5.shapes.add_table(4, 4, Inches(7.1), Inches(1.6), Inches(5.4), Inches(2.6))
    tbl = table_bench.table
    tbl.columns[0].width = Inches(2.1)
    tbl.columns[1].width = Inches(1.1)
    tbl.columns[2].width = Inches(1.1)
    tbl.columns[3].width = Inches(1.1)

    headers = ["Method", "Anchor", "Novel", "Gap"]
    for c, h in enumerate(headers):
        style_cell(tbl.cell(0, c), h, font_size=11, bold=True, color=C_WHITE, bg_color=C_DARK_BLUE)

    rows = [
        ("OmniView (2D ResNet)", "90.7%", "4.3%", "86.4%"),
        ("Extrinsics FiLM (2D)", "83.1%", "24.4%", "58.7%"),
        ("PointNet++ (Ours)", "86.1%", "84.8%", "1.3%"),
    ]
    for r, row in enumerate(rows):
        bg = RGBColor(236, 253, 245) if r == 2 else (C_LIGHT_BG if r % 2 == 1 else C_WHITE)
        color = C_SUCCESS if r == 2 else (C_DANGER if r == 0 else C_NAVY)
        for c, val in enumerate(row):
            style_cell(tbl.cell(r + 1, c), val, font_size=11, bold=(r == 2 or c == 3), color=color, bg_color=bg)

    add_card(
        s5, Inches(7.1), Inches(4.5), Inches(5.4), Inches(2.4),
        "Key Breakthrough Takeaways",
        [
            ("Complete Invariance", "PointNet++ achieves virtually flat performance across all 16 continuous camera angles (86.1% vs 84.8%)."),
            ("85.1% Relative Gain", "Narrows the continuous generalization gap by 85.1 percentage points over state-of-the-art 2D distillation."),
            ("Zero Retraining", "No additional camera viewpoints or data augmentation required.")
        ]
    )

    s5.notes_slide.notes_text_frame.text = (
        "[5:30 - 7:30] This slide shows the primary breakthrough of this thesis. "
        "As seen in the benchmark plot, OmniView in red plunges to 4.3% between anchor angles. "
        "FiLM in orange improves slightly but still exhibits deep troughs. "
        "Our PointNet++ policy in green maintains a flat, continuous line across all 16 viewpoints, scoring 86.1% on anchors and 84.8% on novel in-between angles. "
        "The continuous generalization gap drops from 86.4% to just 1.3%."
    )

    # =========================================================================
    # SLIDE 6: Multi-Task Generalization: 5 Manipulation Archetypes [~2 min]
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Multi-Task Generalization", "Validation Across 5 Diverse Physical Manipulation Archetypes")

    img_m = resolve_img("multitask_7task_viewpoint_generalization_master.png")
    if os.path.exists(img_m):
        s6.shapes.add_picture(img_m, Inches(0.8), Inches(1.6), width=Inches(5.8))

    table_mt = s6.shapes.add_table(6, 4, Inches(6.9), Inches(1.6), Inches(5.6), Inches(3.2))
    tbl_m = table_mt.table
    tbl_m.columns[0].width = Inches(2.3)
    tbl_m.columns[1].width = Inches(1.1)
    tbl_m.columns[2].width = Inches(1.1)
    tbl_m.columns[3].width = Inches(1.1)

    m_headers = ["Manipulation Task", "Anchor", "Novel", "Gap"]
    for c, h in enumerate(m_headers):
        style_cell(tbl_m.cell(0, c), h, font_size=11, bold=True, color=C_WHITE, bg_color=C_DARK_BLUE)

    task_data = [
        ("PickCube-v1 (Free Grasp)", "86.1%", "84.8%", "1.3%"),
        ("PushCube-v1 (Planar)", "89.2%", "84.8%", "4.4%"),
        ("PullCube-v1 (Friction)", "88.3%", "82.5%", "5.8%"),
        ("PokeCube-v1 (Impact)", "78.4%", "72.1%", "6.3%"),
        ("LiftPegUpright-v1 (Torque)", "87.5%", "81.2%", "6.3%"),
    ]
    for r, row in enumerate(task_data):
        bg = C_LIGHT_BG if r % 2 == 1 else C_WHITE
        for c, val in enumerate(row):
            style_cell(tbl_m.cell(r + 1, c), val, font_size=10, bold=(c == 3), color=C_NAVY, bg_color=bg)

    # Summary callout banner below table
    banner = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(5.1), Inches(5.6), Inches(1.8))
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(236, 253, 245)
    banner.line.color.rgb = C_SUCCESS
    tf_b = banner.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "Universal <7% Generalization Gap"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_SUCCESS
    p2 = tf_b.add_paragraph()
    p2.text = "Invariance holds across grasping, sliding, friction tension, impulsive impact, and 6-DoF torque balancing."
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_NAVY
    p2.space_before = Pt(4)

    s6.notes_slide.notes_text_frame.text = (
        "[7:30 - 9:30] We did not stop at pick-and-place. We validated our 3D formulation across five distinct physical manipulation archetypes. "
        "As summarized in the table, across all five tasks—Pick, Push, Pull, Poke, and LiftPegUpright—the generalization gap remains under 7%. "
        "This proves that 3D metric invariance is a universal principle for physical robot manipulation, not an artifact of a single task."
    )

    # =========================================================================
    # SLIDE 7: Real-World Robustness & Edge Deployment [~1.5 min]
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Hardware Resilience & Edge Viability", "Sensor Noise Tolerance & 70.4 Hz Real-Time Edge Control")

    img_r = resolve_img("calibration_noise_robustness_curve.png")
    if os.path.exists(img_r):
        s7.shapes.add_picture(img_r, Inches(0.8), Inches(1.6), width=Inches(5.8))

    add_card(
        s7, Inches(6.9), Inches(1.6), Inches(5.6), Inches(2.6),
        "Sim-to-Real Hardware Resilience",
        [
            ("Calibration Jitter", "Tolerates ±1.5 cm position and ±2.0° angular extrinsics noise with >80% success."),
            ("RealSense Depth Dropout", "Survives up to 20% random depth dropout and structural sensor voids."),
            ("Latency Jitter", "Maintains high task completion despite dynamic 50 ms communication latency.")
        ]
    )

    add_card(
        s7, Inches(6.9), Inches(4.4), Inches(5.6), Inches(2.5),
        "Edge Hardware Deployment Profiling",
        [
            ("NVIDIA RTX 4090", "3.0 ms total latency (333 Hz control capability)."),
            ("NVIDIA RTX 3060", "6.5 ms total latency (154 Hz control capability)."),
            ("Jetson Orin Nano (20W)", "14.2 ms total latency (70.4 Hz inference), comfortably exceeding the 20 Hz robot loop at edge power budget.")
        ]
    )

    s7.notes_slide.notes_text_frame.text = (
        "[9:30 - 11:00] For practical real-world deployment, policies must withstand noisy sensors and run within edge power limits. "
        "Our policy tolerates up to 1.5 cm of calibration jitter and 20% depth sensor dropout without catastrophic degradation. "
        "Furthermore, our hardware profiling on a 20W NVIDIA Jetson Orin Nano shows a total inference time of just 14.2 ms, "
        "delivering 70 Hz throughput for a 20 Hz robot control loop."
    )

    # =========================================================================
    # SLIDE 8: Closed-Loop Video Demo: PickCube-v1 (EMBEDDED VIDEO) [~2 min]
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Closed-Loop Video Demonstration", "16 Simultaneous Closed-Loop Rollouts: PickCube-v1")

    video_pick = os.path.join(REPO_DIR, "btp_submission", "videos", "pickcube_16view_continuous_viewpoint_grid.mp4")
    poster_pick = extract_poster(video_pick, "poster_pickcube.jpg")

    if os.path.exists(video_pick):
        s8.shapes.add_movie(
            video_pick,
            left=Inches(0.8),
            top=Inches(1.6),
            width=Inches(7.2),
            height=Inches(5.2),
            poster_frame_image=poster_pick,
            mime_type="video/mp4"
        )

    add_card(
        s8, Inches(8.3), Inches(1.6), Inches(4.2), Inches(5.2),
        "Demo Technical Highlights",
        [
            ("True Closed-Loop", "In every tile, PointNet++ actively perceives the scene from that camera pose and computes joint commands in real time (NOT video replays)."),
            ("16-View Hemisphere", "Includes 4 trained anchors, 7 novel interpolated viewpoints, 2 extreme 60° flanks, 2 zoom variations, and overhead."),
            ("16/16 Test Score", "100% successful grasp and lift across every viewpoint."),
            ("Inspectability", "Encoded at 5x slowdown (25 FPS) for clear evaluation by the examination committee.")
        ]
    )

    s8.notes_slide.notes_text_frame.text = (
        "[11:00 - 13:00] Now I direct your attention to the video embedded in this slide. "
        "This video demonstrates 16 simultaneous, independent closed-loop test rollouts on PickCube-v1. "
        "Crucially, these are not camera replays of a single rollout—in each tile, the PointNet++ policy actively receives sensory input "
        "from that specific camera angle and computes motor actions in real time. "
        "Notice that whether the camera is at an anchor angle, an unseen interpolated angle, or an extreme flank, the arm smoothly executes the grasp (16 out of 16)."
    )

    # =========================================================================
    # SLIDE 9: Closed-Loop Video Demo: LiftPegUpright-v1 (EMBEDDED VIDEO) [~1.5 min]
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Dexterity Video Demonstration", "Precision 6-DoF Torque Balancing: LiftPegUpright-v1")

    video_peg = os.path.join(REPO_DIR, "btp_submission", "videos", "liftpegupright_16view_continuous_viewpoint_grid.mp4")
    poster_peg = extract_poster(video_peg, "poster_peg.jpg")

    if os.path.exists(video_peg):
        s9.shapes.add_movie(
            video_peg,
            left=Inches(0.8),
            top=Inches(1.6),
            width=Inches(7.2),
            height=Inches(5.2),
            poster_frame_image=poster_peg,
            mime_type="video/mp4"
        )

    add_card(
        s9, Inches(8.3), Inches(1.6), Inches(4.2), Inches(5.2),
        "Complex Dexterity Highlights",
        [
            ("6-DoF Torque Alignment", "The robot grasps an unanchored peg, rotates it against gravity, and seats it upright on the table."),
            ("Novel Perspective Robustness", "Overcomes severe visual foreshortening and self-occlusions across continuous viewpoints."),
            ("16/16 Success Rate", "Flawless task execution across all 16 test camera locations."),
            ("Physics Engine", "Simulated at 100 Hz in PhysX 5 with a 20 Hz policy control rate.")
        ]
    )

    s9.notes_slide.notes_text_frame.text = (
        "[13:00 - 14:30] Next, we demonstrate high-dexterity manipulation on LiftPegUpright-v1. "
        "Here the robot must grasp a lying peg, reorient it in 6-DoF space against gravity, and balance it upright. "
        "Despite severe perspective distortions and partial finger occlusions, the 3D policy successfully completes the task across all 16 angles."
    )

    # =========================================================================
    # SLIDE 10: Academic Rigor: Physical Failure Boundaries [~1.5 min]
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. Empirical Limits & Academic Rigor", "Failure Boundaries on StackCube-v1: Why Single-View 3D Fails")

    # 3 Failure Breakdown Cards
    add_card(
        s10, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "Sub-Phase Failure Funnel",
        [
            ("Reaching Sub-Phase", "86.4% success. Arm approaches the first block reliably."),
            ("Grasp Sub-Phase", "81.2% success. Gripper secures the pick cube firmly."),
            ("Placement Sub-Phase", "0.0% success. Arm drops or knocks over the base block."),
            ("Overall Task Score", "0.0% completion across all continuous camera viewpoints.")
        ]
    )

    add_card(
        s10, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "Bottleneck 1: Self-Occlusion",
        [
            ("Line-of-Sight Block", "When descending to place the upper cube, gripper fingers physically conceal the mating seam."),
            ("Single-View Limit", "No single external camera can perceive the contact interface when the hand obstructs the view."),
            ("Conclusion", "Multi-contact assembly fundamentally demands multi-camera fusion or eye-in-hand tactile sensors.")
        ]
    )

    add_card(
        s10, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "Bottleneck 2: Depth Ambiguity",
        [
            ("Geometry Symmetry", "Uncolored 3D point clouds yield identical geometric signatures for identical cubes."),
            ("Semantic Confusion", "Without semantic segmentation masks, the network confuses top vs bottom cube coordinates during contact."),
            ("Conclusion", "Precision stacking requires semantic instance labeling or colored point clouds.")
        ]
    )

    s10.notes_slide.notes_text_frame.text = (
        "[14:30 - 16:00] In the spirit of scientific rigor, we report negative results on StackCube-v1, where our policy achieved 0.0% success. "
        "Deconstructing the trajectory reveals why: reaching achieves 86.4% and grasping achieves 81.2%, but precision placement fails completely. "
        "This is caused by physical gripper self-occlusion concealing the mating seam, and geometric depth ambiguity between identical uncolored cubes."
    )

    # =========================================================================
    # SLIDE 11: Next-Generation Research Agenda [~1.5 min]
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s10, "10. Future Research Agenda", "Three Post-BTP Architectures for Next-Generation Manipulation")

    add_card(
        s11, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "Architecture A: Self-Calibration",
        [
            ("Online Kinematics", "x_dot_base = J(q) · q_dot tracks end-effector motion via forward kinematics."),
            ("Cross-Attention Flow", "Matches robot velocity with 2D optical flow in pixel space."),
            ("Autonomous Operation", "Recovers extrinsic matrix T_cam^base in the first 5 control steps without manual markers.")
        ]
    )

    add_card(
        s11, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "Architecture B: SE(3) Equivariance",
        [
            ("Steerable Graph Nets", "Enforces mathematical action field equivariance: π(g · P) = g · π(P), ∀g ∈ SO(3)."),
            ("Spherical Harmonics", "Uses tensor products to constrain feature representations."),
            ("Guaranteed Invariance", "Zero performance drop under continuous 3D coordinate rotations.")
        ]
    )

    add_card(
        s11, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "Architecture C: Decoupled Foundation",
        [
            ("Cartesian Wrench", "Predicts operational space impedance commands: F_cmd = K_p · e + K_d · e_dot."),
            ("Local URDF IK", "Delegates joint kinematics to a modular robot model solver."),
            ("Cross-Embodiment", "One visual foundation model controls Franka, xArm, and UR5e without retraining.")
        ]
    )

    s11.notes_slide.notes_text_frame.text = (
        "[16:00 - 17:30] To resolve these physical boundaries, we propose three concrete post-BTP architectures: "
        "Architecture A achieves markerless online self-calibration via forward kinematics and optical flow. "
        "Architecture B incorporates steerable SE(3)-equivariant action fields for guaranteed mathematical rotation invariance. "
        "Architecture C outputs Cartesian impedance wrenches, enabling a single visual foundation policy to control heterogeneous robot arms."
    )

    # =========================================================================
    # SLIDE 12: Conclusion & Acknowledgments (Dark Hero Theme) [~1 min]
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = C_NAVY
    bg12.line.color.rgb = C_NAVY

    # Left: Summary Bullets
    tb12 = s12.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(7.0), Inches(6.0))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    p = tf12.paragraphs[0]
    p.text = "CONCLUSION & KEY CONTRIBUTIONS"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    conclusions = [
        ("Discovered 2D Collapse", "Proved 2D multi-teacher policies suffer an 86.4% performance cliff on continuous viewpoints."),
        ("Formulated 3D Metric Invariance", "Slashed continuous generalization gap on PickCube-v1 from 86.4% down to 1.3%."),
        ("Universal Multi-Task Efficacy", "Validated across 5 physical contact archetypes with <7% generalization gap."),
        ("Sim-to-Real Resilience", "Survives ±1.5 cm / ±2.0° calibration jitter and runs at 70.4 Hz on 20W Jetson Orin Nano."),
        ("Mapped Empirical Boundaries", "Characterized gripper self-occlusion and depth ambiguity limits on StackCube-v1.")
    ]
    for lead, body in conclusions:
        p = tf12.add_paragraph()
        p.space_before = Pt(8)
        p.font.size = Pt(12)
        r_lead = p.add_run()
        r_lead.text = f"• {lead}: "
        r_lead.font.bold = True
        r_lead.font.color.rgb = C_ACCENT
        r_body = p.add_run()
        r_body.text = body
        r_body.font.color.rgb = RGBColor(226, 232, 240)

    # Right: Acknowledgment Card & Closing
    ack_card = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.1), Inches(0.8), Inches(4.5), Inches(4.8))
    ack_card.fill.solid()
    ack_card.fill.fore_color.rgb = C_DARK_BLUE
    ack_card.line.color.rgb = C_ACCENT
    tf_ack = ack_card.text_frame
    tf_ack.word_wrap = True

    p_ah = tf_ack.paragraphs[0]
    p_ah.text = "ACKNOWLEDGMENTS"
    p_ah.font.size = Pt(16)
    p_ah.font.bold = True
    p_ah.font.color.rgb = C_ACCENT

    p_sup = tf_ack.add_paragraph()
    p_sup.text = "Supervisor Guidance:"
    p_sup.font.size = Pt(12)
    p_sup.font.bold = True
    p_sup.font.color.rgb = C_WHITE
    p_sup.space_before = Pt(10)

    p_sup_t = tf_ack.add_paragraph()
    p_sup_t.text = (
        "The author expresses sincere gratitude to his supervisor, Prof. Shivashankar B. Nair, "
        "for his guidance, encouragement, and insightful feedback throughout this thesis."
    )
    p_sup_t.font.size = Pt(11)
    p_sup_t.font.color.rgb = RGBColor(203, 213, 225)
    p_sup_t.space_before = Pt(4)

    p_comp = tf_ack.add_paragraph()
    p_comp.text = "Computational Resources Attribution:"
    p_comp.font.size = Pt(12)
    p_comp.font.bold = True
    p_comp.font.color.rgb = C_WHITE
    p_comp.space_before = Pt(12)

    p_comp_t = tf_ack.add_paragraph()
    p_comp_t.text = (
        "All computational experiments, simulations, and neural network training routines "
        "were conducted entirely on personal GPU hardware without utilizing departmental compute clusters or resources."
    )
    p_comp_t.font.size = Pt(11)
    p_comp_t.font.color.rgb = RGBColor(203, 213, 225)
    p_comp_t.space_before = Pt(4)

    close_box = s12.shapes.add_textbox(Inches(8.1), Inches(5.8), Inches(4.5), Inches(1.2))
    tf_cb = close_box.text_frame
    p_cb = tf_cb.paragraphs[0]
    p_cb.text = "Thank you. Open for questions & demonstration."
    p_cb.font.size = Pt(14)
    p_cb.font.bold = True
    p_cb.font.color.rgb = C_ACCENT

    s12.notes_slide.notes_text_frame.text = (
        "[17:30 - 18:30] In conclusion, this thesis proves why 2D representation learning fails continuous camera interpolation, "
        "demonstrates that 3D metric representations achieve universal spatial invariance across 5 physical manipulation archetypes, "
        "and rigorously maps the empirical boundaries of geometric invariance. "
        "I express my deepest gratitude to Prof. Shivashankar B. Nair for his guidance. "
        "Thank you for your time. I welcome your questions and am ready for interactive demonstration."
    )

    prs.save(OUTPUT_PPTX)
    print(f"[OK] Saved to: {OUTPUT_PPTX}")
    prs.save(ROOT_PPTX)
    print(f"[OK] Saved to: {ROOT_PPTX}")
    prs.save(ARTIFACT_PPTX)
    print(f"[OK] Saved to: {ARTIFACT_PPTX}")

if __name__ == "__main__":
    create_deck()
