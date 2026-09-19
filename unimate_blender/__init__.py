bl_info = {
    "name": "UniMate Motion Generator",
    "author": "Kiran + Codex",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > UniMate",
    "description": "Generate text-conditioned UniMate motion and apply it as a Blender action",
    "category": "Animation",
}

import glob
import json
import os
import re
import subprocess
import sys
import time

import bpy


DEFAULT_ROOT = os.environ.get("UNIMATE_ROOT", "")


def _paths(scene):
    root = bpy.path.abspath(scene.unimate_root)
    python_dir = "Scripts" if os.name == "nt" else "bin"
    python_name = "python.exe" if os.name == "nt" else "python"
    fallback_python = os.path.join(root, ".venv", python_dir, python_name)
    return {
        "root": root,
        "python": bpy.path.abspath(scene.unimate_python) if scene.unimate_python else fallback_python,
        "experiment": bpy.path.abspath(scene.unimate_experiment),
        "model": bpy.path.abspath(scene.unimate_model),
        "cond": bpy.path.abspath(scene.unimate_cond),
        "blender_site": os.path.join(root, "blender_site"),
        "runner": os.path.join(os.path.dirname(__file__), "unimate_safetensors_runner.py"),
    }


def _safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return value[:48] or "motion"


def _active_armature(context):
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    if obj and obj.find_armature():
        return obj.find_armature()
    armatures = [o for o in context.scene.objects if o.type == "ARMATURE"]
    return armatures[0] if len(armatures) == 1 else None


def _generated_action_items(_self, _context):
    actions = sorted(
        (action for action in bpy.data.actions if action.name.startswith("UniMate_")),
        key=lambda action: action.name.lower(),
    )
    if not actions:
        return [("__NONE__", "No UniMate actions", "Generate or import a motion first")]
    return [
        (action.name, action.name, f"Frames {int(action.frame_range[0])}–{int(action.frame_range[1])}")
        for action in actions
    ]


def _install_import_paths(root, blender_site):
    for path in (blender_site, root):
        if path not in sys.path:
            sys.path.insert(0, path)


def _apply_motion(context, motion_path):
    scene = context.scene
    paths = _paths(scene)
    _install_import_paths(paths["root"], paths["blender_site"])

    from Animation import transforms_local
    from data_process.mesh_animation.animate_motion import (
        build_anim_from_npy,
        load_cond_data,
    )
    from data_process.utils.blender_rig import compute_bone_keyframes

    armature = _active_armature(context)
    if armature is None:
        raise RuntimeError("Select an armature, or keep exactly one armature in the scene")

    cond = load_cond_data(paths["cond"], motion_path, "objaverse")
    anim, rest_anim, tpos_global_rot, _ = build_anim_from_npy(
        motion_path, cond, "fk"
    )
    keyframes = compute_bone_keyframes(
        transforms_local(rest_anim)[0],
        transforms_local(anim),
        cond["joint_names"],
        tpos_global_rot,
    )

    action_name = "UniMate_" + _safe_name(scene.unimate_action_name)
    action = bpy.data.actions.new(action_name)
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action

    inserted = 0
    for bone_name, channels in keyframes.items():
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        pose_bone.rotation_mode = "QUATERNION"
        for frame, value in channels["location"]:
            pose_bone.location = value
            pose_bone.keyframe_insert(
                data_path="location", frame=frame + 1, group=bone_name
            )
            inserted += 1
        for frame, value in channels["rotation"]:
            pose_bone.rotation_quaternion = value
            pose_bone.keyframe_insert(
                data_path="rotation_quaternion", frame=frame + 1, group=bone_name
            )
            inserted += 1

    action["unimate_prompt"] = scene.unimate_prompt
    action["unimate_source"] = motion_path
    scene.frame_start = 1
    scene.frame_end = int(len(anim))
    scene.render.fps = 30
    scene.frame_set(1)
    return action, inserted


class UNIMATE_OT_generate(bpy.types.Operator):
    bl_idname = "unimate.generate"
    bl_label = "Generate UniMate Motion"
    bl_description = "Run UniMate inference with the entered prompt"

    _timer = None
    _process = None
    _started = 0.0
    _log_handle = None
    _run_dir = ""

    def execute(self, context):
        scene = context.scene
        paths = _paths(scene)
        required = (paths["python"], paths["experiment"], paths["model"], paths["cond"])
        if not all(os.path.exists(path) for path in required):
            self.report({"ERROR"}, "Check UniMate, experiment, and cond paths")
            return {"CANCELLED"}
        if not scene.unimate_prompt.strip():
            self.report({"ERROR"}, "Enter an animation prompt")
            return {"CANCELLED"}

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._run_dir = os.path.join(paths["root"], "outputs", "blender_plugin", stamp)
        os.makedirs(self._run_dir, exist_ok=True)
        cases_path = os.path.join(self._run_dir, "cases.json")
        case_key = f"{scene.unimate_object_type}-{_safe_name(scene.unimate_action_name)}"
        with open(cases_path, "w", encoding="utf-8") as handle:
            json.dump({case_key: scene.unimate_prompt}, handle, indent=2)

        log_path = os.path.join(self._run_dir, "unimate.log")
        self._log_handle = open(log_path, "w", encoding="utf-8")
        command = [
            paths["python"], paths["runner"],
            "--exp_dir", paths["experiment"],
            "--model_path", paths["model"],
            "--test_cases_json", cases_path,
            "--output_dir", self._run_dir,
            "--num_repetitions", "1",
            "--batch_size", "1",
            "--only_save_motion",
            "--seed", str(scene.unimate_seed),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            command,
            cwd=paths["root"],
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        self._started = time.perf_counter()
        scene.unimate_status = "Generating…"
        scene.unimate_last_output = self._run_dir
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._process.terminate()
            return self._finish(context, cancelled=True)
        if event.type != "TIMER" or self._process.poll() is None:
            return {"PASS_THROUGH"}
        return self._finish(context, cancelled=self._process.returncode != 0)

    def _finish(self, context, cancelled):
        context.window_manager.event_timer_remove(self._timer)
        self._log_handle.close()
        elapsed = time.perf_counter() - self._started
        context.scene.unimate_last_seconds = elapsed
        if cancelled:
            context.scene.unimate_status = "Failed — inspect unimate.log"
            self.report({"ERROR"}, context.scene.unimate_status)
            return {"CANCELLED"}
        motions = sorted(glob.glob(os.path.join(self._run_dir, "motions", "*.npy")))
        if not motions:
            context.scene.unimate_status = "Finished, but no motion file was produced"
            return {"CANCELLED"}
        context.scene.unimate_last_motion = motions[-1]
        context.scene.unimate_status = f"Generated in {elapsed:.1f}s"
        if context.scene.unimate_auto_import:
            try:
                action, _ = _apply_motion(context, motions[-1])
                context.scene.unimate_status += f" — imported {action.name}"
            except Exception as exc:
                context.scene.unimate_status += f" — import failed: {exc}"
        self.report({"INFO"}, context.scene.unimate_status)
        return {"FINISHED"}


class UNIMATE_OT_import_last(bpy.types.Operator):
    bl_idname = "unimate.import_last"
    bl_label = "Import Last Motion"

    def execute(self, context):
        motion = bpy.path.abspath(context.scene.unimate_last_motion)
        if not os.path.isfile(motion):
            self.report({"ERROR"}, "No generated motion file found")
            return {"CANCELLED"}
        try:
            action, inserted = _apply_motion(context, motion)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {action.name} ({inserted} keyed channels)")
        return {"FINISHED"}


class UNIMATE_OT_apply_action(bpy.types.Operator):
    bl_idname = "unimate.apply_action"
    bl_label = "Apply to Character"
    bl_description = "Make the selected UniMate action active on the character armature"

    def execute(self, context):
        scene = context.scene
        action = bpy.data.actions.get(scene.unimate_selected_action)
        armature = _active_armature(context)
        if action is None or armature is None:
            self.report({"ERROR"}, "Select a valid UniMate action and character armature")
            return {"CANCELLED"}
        if armature.animation_data is None:
            armature.animation_data_create()
        armature.animation_data.action = action
        scene.frame_start = max(1, int(action.frame_range[0]))
        scene.frame_end = max(scene.frame_start, int(action.frame_range[1]))
        scene.frame_set(scene.frame_start)
        scene.unimate_status = f"Active: {action.name}"
        self.report({"INFO"}, scene.unimate_status)
        return {"FINISHED"}


class UNIMATE_OT_play_toggle(bpy.types.Operator):
    bl_idname = "unimate.play_toggle"
    bl_label = "Play / Stop"
    bl_description = "Play or stop the active UniMate animation"

    def execute(self, context):
        bpy.ops.screen.animation_play()
        return {"FINISHED"}


class UNIMATE_PT_panel(bpy.types.Panel):
    bl_label = "UniMate"
    bl_idname = "UNIMATE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UniMate"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "unimate_prompt")
        row = layout.row(align=True)
        row.prop(scene, "unimate_action_name")
        row.prop(scene, "unimate_seed")
        layout.prop(scene, "unimate_object_type")
        layout.prop(scene, "unimate_auto_import")
        layout.operator("unimate.generate", icon="PLAY")
        layout.operator("unimate.import_last", icon="ACTION")

        actions = layout.box()
        actions.label(text="Generated Animations", icon="ACTION")
        actions.prop(scene, "unimate_selected_action", text="")
        actions.operator("unimate.apply_action", icon="ARMATURE_DATA")
        row = actions.row(align=True)
        row.operator("unimate.play_toggle", text="Play / Stop", icon="PLAY")
        row.operator("screen.frame_jump", text="First", icon="REW").end = False
        active = _active_armature(context)
        if active and active.animation_data and active.animation_data.action:
            actions.label(text=f"Playing: {active.animation_data.action.name}")
        box = layout.box()
        box.label(text=scene.unimate_status)
        if scene.unimate_last_seconds:
            box.label(text=f"Last run: {scene.unimate_last_seconds:.1f} seconds")
        advanced = layout.box()
        advanced.label(text="Paths")
        advanced.prop(scene, "unimate_root")
        advanced.prop(scene, "unimate_python")
        advanced.prop(scene, "unimate_experiment")
        advanced.prop(scene, "unimate_model")
        advanced.prop(scene, "unimate_cond")


CLASSES = (
    UNIMATE_OT_generate,
    UNIMATE_OT_import_last,
    UNIMATE_OT_apply_action,
    UNIMATE_OT_play_toggle,
    UNIMATE_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.unimate_prompt = bpy.props.StringProperty(
        name="Prompt", default="A child waves happily and takes two steps forward"
    )
    bpy.types.Scene.unimate_action_name = bpy.props.StringProperty(
        name="Action", default="Generated_Motion"
    )
    bpy.types.Scene.unimate_seed = bpy.props.IntProperty(name="Seed", default=10, min=0)
    bpy.types.Scene.unimate_object_type = bpy.props.StringProperty(
        name="Skeleton", default="littleKrishna"
    )
    bpy.types.Scene.unimate_auto_import = bpy.props.BoolProperty(
        name="Import automatically", default=True
    )
    bpy.types.Scene.unimate_root = bpy.props.StringProperty(
        name="UniMate Root", subtype="DIR_PATH", default=DEFAULT_ROOT
    )
    bpy.types.Scene.unimate_python = bpy.props.StringProperty(
        name="Python", subtype="FILE_PATH", default=""
    )
    bpy.types.Scene.unimate_experiment = bpy.props.StringProperty(
        name="Experiment", subtype="DIR_PATH",
        default=(os.path.join(DEFAULT_ROOT, "outputs", "littleKrishna_unimate")
                 if DEFAULT_ROOT else ""),
    )
    bpy.types.Scene.unimate_model = bpy.props.StringProperty(
        name="SafeTensors", subtype="FILE_PATH",
        default=(os.path.join(
            DEFAULT_ROOT, "outputs", "uniml3d_60frames_graph_adaln",
            "model_ema.safetensors",
        ) if DEFAULT_ROOT else ""),
    )
    bpy.types.Scene.unimate_cond = bpy.props.StringProperty(
        name="Conditioning", subtype="FILE_PATH",
        default=(os.path.join(DEFAULT_ROOT, "dataset", "features", "custom", "cond.npy")
                 if DEFAULT_ROOT else ""),
    )
    bpy.types.Scene.unimate_last_motion = bpy.props.StringProperty(
        name="Last Motion", subtype="FILE_PATH"
    )
    bpy.types.Scene.unimate_last_output = bpy.props.StringProperty(name="Last Output")
    bpy.types.Scene.unimate_last_seconds = bpy.props.FloatProperty(name="Last Seconds")
    bpy.types.Scene.unimate_status = bpy.props.StringProperty(name="Status", default="Ready")
    bpy.types.Scene.unimate_selected_action = bpy.props.EnumProperty(
        name="Generated Animation",
        description="UniMate action to apply to the character",
        items=_generated_action_items,
    )


def unregister():
    for name in (
        "unimate_prompt", "unimate_action_name", "unimate_seed",
        "unimate_object_type", "unimate_auto_import", "unimate_root", "unimate_python",
        "unimate_experiment", "unimate_model", "unimate_cond", "unimate_last_motion",
        "unimate_last_output", "unimate_last_seconds", "unimate_status",
        "unimate_selected_action",
    ):
        delattr(bpy.types.Scene, name)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
