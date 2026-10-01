#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fps24_gui.py — 30帧(24帧素材复制帧) → 24帧 批量还原工具 (GUI版)

要求: Windows/Linux + NVIDIA显卡 (HEVC NVENC), Python3, ffmpeg 在 PATH 中
"""

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

def _find_bin(name):
    """优先使用exe同目录/PyInstaller内置目录中的ffmpeg，否则回退PATH"""
    exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    candidates = [os.path.join(exe_dir, name),
                  os.path.join(getattr(sys, "_MEIPASS", ""), name)]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return name  # PATH

FFMPEG = _find_bin("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
FFPROBE = _find_bin("ffprobe.exe" if os.name == "nt" else "ffprobe")

# ---------------- 核心处理逻辑 ----------------

def have_nvenc():
    """检测NVENC是否真正可用（列出编码器≠显卡存在，需试编码一帧）"""
    try:
        r = subprocess.run(
            [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=size=64x64:duration=0.1",
             "-c:v", "hevc_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False

def probe(path):
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
         "-of", "json", path], capture_output=True, text=True, check=True)
    import json
    s = json.loads(r.stdout)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return float(num) / float(den), int(s.get("nb_frames") or 0), float(s.get("duration") or 0)

def frame_diffs(path, thumb_w=256):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    src_w, src_h = map(int, r.stdout.strip().splitlines()[0].split(","))
    thumb_h = max(2, int(round(src_h * thumb_w / src_w / 2) * 2))
    frame_size = thumb_w * thumb_h
    cmd = [FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={thumb_w}:-2",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)
    diffs, prev = [], None
    while True:
        buf = p.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    p.stdout.close()
    p.wait()
    return np.array(diffs)

def detect_phases(diffs, period=5):
    """默认帧复制模式: 切镜分段 + 每段找复制帧相位。返回(分段列表, 是否可信)"""
    med = np.median(diffs[diffs > 0]) if (diffs > 0).any() else 1.0
    p95 = np.percentile(diffs, 95)
    cut_thr = max(10.0 * med, 5.0 * p95, 8.0)
    cuts = [0] + [int(j) + 1 for j in np.where(diffs > cut_thr)[0]] + [len(diffs) + 1]

    segments, global_votes = [], []
    for si in range(len(cuts) - 1):
        start, end = cuts[si], cuts[si + 1]
        if end - start < period * 2:
            segments.append((start, end, None))
            continue
        seg = diffs[start:end - 1]
        smed = np.median(seg[seg > 0]) if (seg > 0).any() else med
        dup = np.where(seg < 0.4 * smed)[0]
        if len(dup) < max(2, (end - start) // period // 4):
            segments.append((start, end, None))
            continue
        votes = (dup + 1) % period
        segments.append((start, end, int(np.bincount(votes, minlength=period).argmax())))
        global_votes += list(votes)

    if not global_votes:
        return [], False  # 全片找不到复制帧特征，不像24->30复制
    fallback = int(np.bincount(global_votes, minlength=period).argmax())
    return [(s, e, (ph if ph is not None else fallback)) for s, e, ph in segments], True

def build_select(segments):
    return "+".join(
        f"between(n,{s},{e-1})*not(eq(mod(n,5),{ph}))" for s, e, ph in segments)

def process_file(input_path, output_path, src_fps, cq, use_nvenc, progress_cb, log_cb):
    """处理单个文件。返回 (成功, 信息)"""
    fps, nb_frames, duration = probe(input_path)
    log_cb(f"[分析] {os.path.basename(input_path)}: {fps:.3f}fps, {nb_frames}帧, {duration:.1f}秒")
    if abs(fps - 30) > 1.5 and abs(fps - 30000 / 1001) > 1.5:
        return False, f"输入不是30fps视频 ({fps:.2f}fps)，已跳过"

    diffs = frame_diffs(input_path)
    progress_cb(10)
    segments, ok = detect_phases(diffs)
    if not ok:
        return False, "未检测到复制帧特征，不像24帧素材误导出为30帧，已跳过"
    n_drop = sum((e - s) // 5 for s, e, _ in segments)
    log_cb(f"[检测] {len(segments)}个镜头分段, 剔除约{n_drop}帧")
    for s, e, ph in segments:
        log_cb(f"       帧[{s:>6},{e:>6}) 相位={ph}")

    out_fps = "24000/1001" if src_fps == "23.976" else "24"
    vf = f"select='{build_select(segments)}',setpts=N/{out_fps}/TB"
    if use_nvenc:
        vcodec = ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(cq)]
    else:
        vcodec = ["-c:v", "libx265", "-crf", str(cq), "-preset", "medium", "-tag:v", "hvc1"]
    cmd = [FFMPEG, "-y", "-v", "info", "-nostats", "-i", input_path,
           "-vf", vf, "-r", out_fps] + vcodec + ["-c:a", "copy",
           "-progress", "pipe:1", output_path]

    log_cb("[转码] " + ("HEVC NVENC" if use_nvenc else "CPU libx265"))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    total_ms = duration * 1000
    for line in p.stdout:
        if line.startswith("out_time_ms="):
            try:
                t = int(line.strip().split("=")[1])
                progress_cb(10 + min(89, int(89 * t / max(total_ms, 1))))
            except ValueError:
                pass
    _, err = p.communicate()
    if p.returncode != 0:
        return False, "ffmpeg转码失败: " + err[-400:]
    progress_cb(99)

    fps_o, nb_o, dur_o = probe(output_path)
    log_cb(f"[完成] 输出 {fps_o:.3f}fps, {nb_o}帧, 时长差{abs(dur_o-duration):.3f}s")
    if abs(dur_o - duration) > 0.05:
        return False, f"输出时长异常 ({dur_o:.2f}s vs {duration:.2f}s)"
    return True, "OK"

# ---------------- GUI ----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("30帧→24帧 复制帧剔除工具  (24fps素材误导出为30fps的修复)")
        self.geometry("760x600")
        self.q = queue.Queue()
        self.worker = None

        frm_top = ttk.Frame(self); frm_top.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm_top, text="文件列表 (支持批量):").pack(anchor="w")
        frm_list = ttk.Frame(frm_top); frm_list.pack(fill="x", pady=4)
        self.listbox = tk.Listbox(frm_list, height=8, selectmode=tk.EXTENDED)
        sb = ttk.Scrollbar(frm_list, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="x", expand=True); sb.pack(side="right", fill="y")
        frm_btns = ttk.Frame(frm_top); frm_btns.pack(fill="x", pady=2)
        ttk.Button(frm_btns, text="添加文件", command=self.add_files).pack(side="left")
        ttk.Button(frm_btns, text="移除选中", command=self.remove_sel).pack(side="left", padx=6)
        ttk.Button(frm_btns, text="清空", command=lambda: self.listbox.delete(0, "end")).pack(side="left")

        frm_opt = ttk.LabelFrame(self, text="选项"); frm_opt.pack(fill="x", padx=10, pady=6)
        row1 = ttk.Frame(frm_opt); row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(row1, text="素材真实帧率:").pack(side="left")
        self.src_fps = tk.StringVar(value="24")
        ttk.Radiobutton(row1, text="24", variable=self.src_fps, value="24").pack(side="left", padx=4)
        ttk.Radiobutton(row1, text="23.976", variable=self.src_fps, value="23.976").pack(side="left")
        ttk.Label(row1, text="   画质 CQ/CRF:").pack(side="left")
        self.cq = tk.IntVar(value=24)
        ttk.Scale(row1, from_=18, to=32, variable=self.cq, orient="horizontal",
                  length=120).pack(side="left")
        ttk.Label(row1, textvariable=self.cq, width=3).pack(side="left")
        row2 = ttk.Frame(frm_opt); row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="输出目录:").pack(side="left")
        self.outdir = tk.StringVar(value="(与源文件同目录)")
        ttk.Entry(row2, textvariable=self.outdir, width=46).pack(side="left", padx=4)
        ttk.Button(row2, text="浏览…", command=self.browse_outdir).pack(side="left")
        row3 = ttk.Frame(frm_opt); row3.pack(fill="x", padx=8, pady=4)
        self.fallback = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="无N卡时自动回退CPU编码 (libx265, 较慢)",
                        variable=self.fallback).pack(side="left")

        frm_run = ttk.Frame(self); frm_run.pack(fill="x", padx=10)
        self.btn_start = ttk.Button(frm_run, text="开始处理", command=self.start)
        self.btn_start.pack(side="left")
        self.status = ttk.Label(frm_run, text="就绪")
        self.status.pack(side="left", padx=10)
        self.pbar = ttk.Progressbar(frm_run, length=220, mode="determinate")
        self.pbar.pack(side="right")

        self.log = tk.Text(self, height=14, state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        self.log.pack(fill="both", expand=True, padx=10, pady=8)
        self.log.tag_config("ok", foreground="#6fcf6f")
        self.log.tag_config("err", foreground="#e06c6c")

        self.nvenc_ok = have_nvenc()
        self.log_msg(f"NVENC检测: {'可用 ✅' if self.nvenc_ok else '不可用 ❌ (将回退CPU)'}\n")
        self.after(100, self.poll)

    def add_files(self):
        for f in filedialog.askopenfilenames(
                filetypes=[("视频文件", "*.mp4 *.mov *.mkv *.avi *.mts *.m4v"), ("所有文件", "*.*")]):
            self.listbox.insert("end", f)

    def remove_sel(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)

    def browse_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.outdir.set(d)

    def log_msg(self, msg, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n", tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        files = list(self.listbox.get(0, "end"))
        if not files:
            messagebox.showwarning("提示", "请先添加文件")
            return
        if not self.nvenc_ok and not self.fallback.get():
            messagebox.showerror("错误", "未检测到NVIDIA NVENC，且未允许CPU回退")
            return
        self.btn_start.configure(state="disabled")
        self.worker = threading.Thread(target=self.run_all, args=(files,), daemon=True)
        self.worker.start()

    def run_all(self, files):
        outdir = self.outdir.get()
        use_nvenc = self.nvenc_ok
        ok_cnt, fail_cnt = 0, 0
        for idx, f in enumerate(files):
            stem, ext = os.path.splitext(f)
            od = outdir if (outdir and outdir != "(与源文件同目录)") else os.path.dirname(f)
            out = os.path.join(od, os.path.basename(stem) + "_24fps" + (ext or ".mp4"))
            self.q.put(("status", f"({idx+1}/{len(files)}) {os.path.basename(f)}"))
            try:
                ok, msg = process_file(
                    f, out, self.src_fps.get(), self.cq.get(), use_nvenc,
                    progress_cb=lambda v: self.q.put(("prog", v)),
                    log_cb=lambda m: self.q.put(("log", m)))
            except Exception as e:
                ok, msg = False, f"异常: {e}"
            if ok:
                ok_cnt += 1
                self.q.put(("log", f"✅ 成功 → {out}", "ok"))
            else:
                fail_cnt += 1
                self.q.put(("log", f"❌ 失败: {msg}", "err"))
            self.q.put(("prog", 0))
        self.q.put(("log", f"\n全部完成: 成功{ok_cnt} 失败{fail_cnt}", "ok" if fail_cnt == 0 else "err"))
        self.q.put(("status", "就绪"))
        self.q.put(("done", None))

    def poll(self):
        try:
            while True:
                kind, *payload = self.q.get_nowait()
                if kind == "log":
                    self.log_msg(payload[0], payload[1] if len(payload) > 1 else None)
                elif kind == "prog":
                    self.pbar["value"] = payload[0]
                elif kind == "status":
                    self.status.configure(text=payload[0])
                elif kind == "done":
                    self.btn_start.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self.poll)

if __name__ == "__main__":
    App().mainloop()
