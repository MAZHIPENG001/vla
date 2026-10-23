// 检测结果、区间选择和选择清单分别管理；不修改或导出源数据集。
import {$, state, api, pretty, error, run, cell, button, download, findings, anomalyStyles, format} from './common.js';

const labels = {passed: '已检查', flagged: '局部异常', review: '含未验证项', clear: '未标记异常',
    unreadable: '不可检查', unverified: '未验证', disabled: '未启用', suspect: '疑似异常', waiting: '正常等待'};
const signalFields = [
    ['kind', '信号含义'], ['dimensions', '参与维度（空=全部）'],
    ['static_epsilon', '静止容差（支持逗号向量）'], ['static_min_frames', '长静止：连续帧数（可空）'],
    ['static_min_seconds', '长静止：真实持续秒数（可空）'], ['velocity_zscore', '速度 Z-score（可空）'],
    ['acceleration_zscore', '加速度 Z-score（可空）'], ['jerk_limit', 'Jerk 上限（空=未校准）'],
];
const imageFields = [
    ['inspect_images', '检查图像'], ['image_stride', '图像采样步长'], ['image_width', '指标计算宽度'],
    ['blur_variance_min', '清晰度方差下限（空=未校准）'], ['max_bad_image_ratio', '图像异常比例参考（仅统计）'],
    ['uniform_std_max', '低纹理标准差上限'], ['saturation_fraction', '近黑 / 白像素比例'],
    ['repeated_run', '连续重复图像数'], ['frame_gap_factor', '采样缺口倍数（相对真实间隔中位数）'],
    ['sync_tolerance_seconds', '同步容差（秒）'], ['state_timestamp_key', '状态时间戳字段'],
];
let reportPolicy = null, jobId = null, running = false, revision = -1, page = 0, pollTimer;
const pendingDetails = new Map();
let handlers;
const report = () => state.qualityReport;

function setting(parent, id, title, value, kind) {
    const label = document.createElement('label');
    label.textContent = title;
    const input = document.createElement(kind === 'kind' ? 'select' : 'input');
    input.id = id;
    if (kind === 'kind') {
        for (const name of ['position', 'command', 'velocity', 'acceleration']) {
            const option = document.createElement('option');
            option.value = name; option.textContent = name; input.append(option);
        }
    } else input.type = typeof value === 'boolean' ? 'checkbox' : 'text';
    if (typeof value === 'boolean') input.checked = value;
    else input.value = value == null ? '' : Array.isArray(value) ? value.join(',') : value;
    label.append(input); parent.append(label);
}

function buildSettings() {
    for (const role of ['state', 'action']) {
        const group = document.createElement('div'); group.className = 'signal-group';
        const heading = document.createElement('h3');
        heading.textContent = `${role.toUpperCase()} · ${state.info.schema[role].key || '未映射'}`;
        group.append(heading);
        for (const [key, title] of signalFields) setting(group, `${role}_${key}`, title, state.info.policy[role][key], key);
        $('signalSettings').append(group);
    }
    for (const [key, title] of imageFields) setting($('imageSettings'), key, title, state.info.policy[key]);
    $('cameraClocks').value = JSON.stringify(state.info.policy.camera_timestamp_keys);
}

function numberValue(text, optional = false) {
    if (optional && text.trim() === '') return null;
    if (text.trim() === '') throw new Error('必填数值不能为空');
    const value = Number(text);
    if (!Number.isFinite(value)) throw new Error(`无效数值：${text}`);
    return value;
}

function policy() {
    const result = {};
    for (const role of ['state', 'action']) {
        const rule = {};
        for (const [key] of signalFields) {
            const value = $(`${role}_${key}`).value.trim();
            if (key === 'kind') rule[key] = value;
            else if (key === 'dimensions') rule[key] = value ? value.split(',').map(x => numberValue(x)) : null;
            else if (['static_epsilon', 'jerk_limit'].includes(key) && value.includes(',')) rule[key] = value.split(',').map(x => numberValue(x));
            else rule[key] = numberValue(value, key !== 'static_epsilon');
        }
        result[role] = rule;
    }
    for (const [key] of imageFields) result[key] = key === 'inspect_images' ? $(key).checked
        : key === 'state_timestamp_key' ? $(key).value : numberValue($(key).value, key === 'blur_variance_min');
    result.camera_timestamp_keys = JSON.parse($('cameraClocks').value);
    return result;
}

function updatePreviewButtons() {
    const disabled = !report()?.complete || state.qualityStale;
    $('previewNormal').disabled = disabled;
    $('previewSuspect').disabled = disabled;
    $('downloadSelection').disabled = !report() || state.frameRanges === null;
}

async function startQuality(event) {
    event.preventDefault();
    if (running) return;
    const currentPolicy = policy(), scope = $('qualityScope').value;
    let ids;
    if (scope === 'all') ids = Object.keys(state.info.episodes).map(Number);
    else if (scope === 'task') {
        if (state.task === null) throw new Error('请先选择具体 Task');
        ids = state.info.tasks.find(task => task.name === state.task).episode_ids;
    } else {
        if (!state.episode) throw new Error('请先选择 Episode');
        ids = [state.episode.index];
    }
    $('runQuality').disabled = true;
    try {
        const job = await api('/api/quality/start', {policy: currentPolicy, episode_ids: ids, output_root: $('reportRoot').value});
        reportPolicy = JSON.stringify(currentPolicy); jobId = job.id; running = true; revision = -1;
        $('reportSavedPath').value = job.report_path;
    } catch (err) { $('runQuality').disabled = false; throw err; }
    state.qualityReport = null; state.selectedSegment = null; state.qualityStale = false;
    state.qualityDetails.clear(); pendingDetails.clear(); clearTimeout(pollTimer);
    handlers.stopPlayback();
    if (state.frameRanges !== null) await handlers.setFilter(null);
    $('stopQuality').disabled = false; $('downloadReport').disabled = true;
    updatePreviewButtons();
    for (const id of ['qualityRows', 'typeStatistics', 'segmentRows']) $(id).replaceChildren();
    $('segmentPage').textContent = '';
    $('qualityDetail').textContent = '正在检测…';
    $('qualityProgress').max = ids.length; $('qualityProgress').value = 0;
    renderAnnotations(); handlers.drawChart(); await pollQuality();
}

async function pollQuality() {
    if (!jobId) return;
    const requestedJob = jobId;
    try {
        const data = await api(`/api/quality/status?id=${encodeURIComponent(jobId)}&since=${revision}`);
        if (requestedJob !== jobId) return;
        running = data.status === 'running';
        const rows = new Map(data.reset ? [] : (report()?.episodes || []).map(row => [row.episode_index, row]));
        const existing = new Set(data.episode_ids);
        for (const id of rows.keys()) if (!existing.has(id)) { rows.delete(id); state.qualityDetails.delete(id); }
        for (const row of data.report.episodes) {
            rows.set(row.episode_index, row);
            if (state.qualityDetails.get(row.episode_index)?.revision !== row.revision) state.qualityDetails.delete(row.episode_index);
        }
        state.qualityReport = {...data.report, episodes: [...rows.values()]};
        revision = data.revision;
        $('qualityProgress').max = data.total;
        $('qualityProgress').value = data.completed;
        $('reportSavedPath').value = data.report_path;
        $('runQuality').disabled = running; $('stopQuality').disabled = !running; $('importReport').disabled = running;
        try { state.qualityStale = JSON.stringify(policy()) !== reportPolicy; } catch (_) { state.qualityStale = true; }
        renderReport(); renderAnnotations(); handlers.drawChart();
        if (running) pollTimer = setTimeout(pollQuality, 500);
        else {
            await refreshHistory();
            if (data.status === 'error') throw new Error(data.error);
        }
    } catch (err) {
        if (requestedJob !== jobId) return;
        $('qualityStatus').textContent = `检测结果读取失败：${err.message}`; error(err);
        // 网络暂时中断时继续取增量，不丢弃已显示的标记。
        if (running) pollTimer = setTimeout(pollQuality, 1500);
    }
}

async function refreshHistory() {
    const data = await api('/api/quality/history');
    const selected = $('reportHistory').value;
    $('reportHistory').replaceChildren(new Option('选择历史检测…', ''));
    for (const item of data.reports) $('reportHistory').append(new Option(
        `${item.created_at} · ${item.checked_episodes} 条 · ${item.complete ? '完整' : '部分'}`, item.path));
    $('reportHistory').value = selected;
}

function applyPolicy(value) {
    for (const role of ['state', 'action']) for (const [key] of signalFields) {
        const v = value[role][key];
        $(`${role}_${key}`).value = v == null ? '' : Array.isArray(v) ? v.join(',') : v;
    }
    for (const [key] of imageFields) {
        if (typeof value[key] === 'boolean') $(key).checked = value[key];
        else $(key).value = value[key] ?? '';
    }
    $('cameraClocks').value = JSON.stringify(value.camera_timestamp_keys);
    reportPolicy = JSON.stringify(policy());
    state.qualityStale = false;
}

async function activate(data) {
    clearTimeout(pollTimer); jobId = data.id; revision = -1;
    state.qualityReport = null; state.qualityDetails.clear(); pendingDetails.clear(); state.selectedSegment = null;
    applyPolicy(data.policy);
    await pollQuality();
}

export async function restoreQuality() {
    $('reportRoot').value = state.info.report_root;
    const data = await api('/api/quality/active');
    if (data.id) await activate(data);
    else if (data.warning) $('qualityStatus').textContent = `历史结果未恢复：${data.warning}。可选择其他历史报告或重新检测。`;
    await refreshHistory();
}

async function ensureDetails(eid) {
    const summary = report()?.episodes.find(row => row.episode_index === eid);
    if (!summary || !jobId) return null;
    const cached = state.qualityDetails.get(eid);
    if (cached?.revision === summary.revision) return cached.row;
    const key = `${jobId}:${eid}:${summary.revision}`;
    if (pendingDetails.has(key)) return pendingDetails.get(key);
    const requestedJob = jobId;
    const promise = api(`/api/quality/episode?id=${encodeURIComponent(jobId)}&episode=${eid}`).then(row => {
        if (requestedJob !== jobId || report()?.episodes.find(x => x.episode_index === eid)?.revision !== summary.revision) return null;
        state.qualityDetails.delete(eid);
        state.qualityDetails.set(eid, {revision: summary.revision, row});
        if (state.qualityDetails.size > 3) state.qualityDetails.delete(state.qualityDetails.keys().next().value);
        return row;
    }).finally(() => pendingDetails.delete(key));
    pendingDetails.set(key, promise);
    return promise;
}

function renderReport() {
    const summary = report().summary;
    $('qualityStatus').textContent = `${running ? '正在检测（暂存结果持续更新）' : report().complete ? '检测完成' : '部分检测报告'}：${summary.checked_episodes}/${summary.total_requested} 条 Episode · 未标记 ${summary.normal_frames} 帧 · 疑似异常 ${summary.suspect_frames} 帧 / ${summary.anomaly_segments} 个区间 · 不可检查 ${summary.unchecked_frames} 帧 · 正常等待 ${summary.waiting_segments} 个区间`;
    $('qualityLiveBanner').hidden = false;
    $('qualityLiveText').textContent = $('qualityStatus').textContent;
    $('downloadReport').disabled = running; updatePreviewButtons();
    for (const element of document.querySelectorAll('#episodes [data-episode]')) {
        const row = findings(Number(element.dataset.episode));
        element.querySelector('.episode-quality').textContent = row ? `疑似异常 ${row.suspect_frames} 帧${row.episode_complete === false ? ' · 检测中' : ''}` : '';
    }
    $('typeStatistics').replaceChildren();
    for (const stage of summary.by_type) {
        const row = document.createElement('tr');
        for (const key of ['stage', 'segments', 'affected_frames', 'affected_episodes']) cell(row, stage[key]);
        $('typeStatistics').append(row);
    }
    const selectedEpisode = $('segmentEpisode').value;
    $('qualityRows').replaceChildren(); $('segmentEpisode').replaceChildren();
    const option = document.createElement('option'); option.value = 'all'; option.textContent = '全部 Episode'; $('segmentEpisode').append(option);
    for (const result of report().episodes) {
        const row = document.createElement('tr');
        for (const value of [result.episode_index, result.tasks.join('; '), result.length, result.normal_frames,
            result.suspect_frames, result.unchecked_frames, result.segments.filter(s => s.status === 'suspect').length]) cell(row, value);
        cell(row, result.episode_complete === false ? `检测中 · ${result.phase || ''}` : labels[result.status]);
        row.onclick = run(async () => {
            $('qualityDetail').textContent = pretty(await ensureDetails(result.episode_index) || result);
            const segment = result.segments.find(s => s.status === 'suspect');
            if (segment) await selectSegment(segment);
            else { $('qualityDialog').close(); await handlers.openEpisode(result.episode_index, 0); }
        });
        $('qualityRows').append(row);
        const option = document.createElement('option'); option.value = result.episode_index;
        option.textContent = `Episode ${result.episode_index}`; $('segmentEpisode').append(option);
    }
    $('segmentEpisode').value = selectedEpisode;
    if (!$('segmentEpisode').value) $('segmentEpisode').value = 'all';
    renderSegments();
}

function renderSegments() {
    const segments = (report()?.episodes || []).flatMap(row => row.segments).filter(segment =>
        ($('segmentType').value === 'all' || segment.type === $('segmentType').value)
        && ($('segmentEpisode').value === 'all' || segment.episode_index === Number($('segmentEpisode').value))
        && (segment.status !== 'waiting' || $('includeWaiting').checked));
    page = Math.max(0, Math.min(page, Math.ceil(segments.length / 100) - 1));
    $('segmentRows').replaceChildren();
    for (const segment of segments.slice(page * 100, (page + 1) * 100)) {
        const row = document.createElement('tr');
        cell(row, segment.episode_index);
        cell(row, `${anomalyStyles[segment.type].label} · ${segment.role || segment.check}`);
        cell(row, `${segment.start_frame}–${segment.end_frame}`);
        cell(row, `${format(segment.start_timestamp)}–${format(segment.end_timestamp)}`);
        cell(row, format(segment.duration_seconds));
        cell(row, segment.dimensions.join(', ') || '—');
        cell(row, labels[segment.status]);
        row.onclick = run(() => selectSegment(segment)); $('segmentRows').append(row);
    }
    $('segmentPage').textContent = `${segments.length} 个区间 · 第 ${segments.length ? page + 1 : 0} / ${Math.ceil(segments.length / 100)} 页`;
    $('segmentPrevious').disabled = page === 0;
    $('segmentNext').disabled = (page + 1) * 100 >= segments.length;
}

export async function selectSegment(segment, navigate = true) {
    const detail = await ensureDetails(segment.episode_index);
    segment = detail?.segments.find(item => item.id === segment.id) || segment;
    state.selectedSegment = segment;
    $('qualityDetail').textContent = pretty(segment);
    if (navigate) {
        if ($('qualityDialog').open) $('qualityDialog').close();
        await handlers.openEpisode(segment.episode_index, segment.start_frame, segment.end_frame);
    }
    renderAnnotations(); handlers.drawChart();
    $('selectedSegmentDetails').open = true;
}

export function renderAnnotations() {
    updatePreviewButtons();
    const row = findings(state.episode?.index);
    $('anomalyReview').hidden = !row;
    if (!row) return;
    const index = report()?.episodes.find(item => item.episode_index === row.episode_index);
    if (!state.qualityDetails.has(row.episode_index)) {
        const requested = row.episode_index;
        ensureDetails(requested).then(detail => {
            if (detail && state.episode?.index === requested) { renderAnnotations(); handlers.drawChart(); }
        }).catch(error);
    }
    $('anomalySummary').textContent = `${state.qualityStale ? '阈值已改变，以下为上次检测标记，需重新检测。' : ''}${index.episode_complete === false ? '正在检查本条轨迹，暂存标记持续更新。' : report().complete ? '' : '部分检测报告。'}${row.normal_frames} 帧未标记 · ${row.suspect_frames} 帧疑似异常 · ${row.unchecked_frames} 帧不可检查 · ${row.unverified_checks.length} 项未验证 / 未启用`;
    const flags = row.frame_flags?.find(item => item.frame_index === state.frame);
    const types = [...new Set(row.segments.filter(s => s.status === 'suspect' && s.start_frame <= state.frame && state.frame <= s.end_frame).map(s => s.type))];
    $('currentFrameFlags').textContent = flags ? `当前帧标记：${flags.types.map(type => anomalyStyles[type].label).join(' / ')} · 原始 frame_index ${format(flags.source_frame_index)} · timestamp ${format(flags.timestamp)}` : types.length ? `当前帧标记：${types.map(type => anomalyStyles[type].label).join(' / ')} · 正在读取完整明细` : '当前帧没有被本轮已执行检查标记；未验证项仍需复核。';
    const intervals = row.segments.filter(s => state.anomalyTypes.has(s.type) && (s.status !== 'waiting' || state.showWaiting));
    $('anomalyIntervals').replaceChildren();
    for (const segment of intervals) {
        const entry = button(`${anomalyStyles[segment.type].label} · ${segment.role || segment.check} · ${segment.start_frame}–${segment.end_frame}${segment.status === 'waiting' ? '（等待）' : ''}`, () => selectSegment(segment));
        entry.style.color = anomalyStyles[segment.type].color;
        entry.dataset.segmentId = segment.id;
        $('anomalyIntervals').append(entry);
    }
    const selected = state.selectedSegment;
    $('selectedSegmentDetails').hidden = !selected || selected.episode_index !== row.episode_index;
    if (selected && selected.episode_index === row.episode_index) {
        $('selectedSegmentTitle').textContent = `${anomalyStyles[selected.type].label} · ${selected.role || selected.check} · 帧 ${selected.start_frame}–${selected.end_frame} · ${format(selected.start_timestamp)}–${format(selected.end_timestamp)} s · ${labels[selected.status]}（展开查看数值 / 阈值 / 差分窗口）`;
        $('selectedSegmentData').textContent = pretty(selected);
    }
}

async function preview(mode) {
    if (!report()?.complete || state.qualityStale) throw new Error('请先完成当前参数的检测');
    const ranges = {};
    for (const row of report().episodes) {
        const segments = mode === 'normal' ? row.normal_segments : row.suspect_segments;
        if (segments.length) ranges[row.episode_index] = segments.map(s => [s.start_frame, s.end_frame]);
    }
    $('qualityDialog').close();
    await handlers.setFrameFilter(ranges, mode); renderAnnotations(); handlers.drawChart();
}

export function bindQuality(callbacks) {
    handlers = callbacks; buildSettings();
    $('qualityButton').onclick = () => { $('qualityDialog').showModal(); };
    $('qualityLiveOpen').onclick = () => { $('qualityDialog').showModal(); };
    $('refreshHistory').onclick = run(refreshHistory);
    $('reportHistory').onchange = () => { if ($('reportHistory').value) $('reportImportPath').value = $('reportHistory').value; };
    $('importReport').onclick = run(async () => {
        const data = await api('/api/quality/import', {path: $('reportImportPath').value});
        handlers.stopPlayback();
        if (state.frameRanges !== null) await handlers.setFilter(null);
        await activate(data);
    });
    $('qualityForm').onsubmit = run(startQuality);
    $('qualityForm').oninput = run(async () => {
        if (!report()) return;
        try { state.qualityStale = JSON.stringify(policy()) !== reportPolicy; } catch (_) { state.qualityStale = true; }
        updatePreviewButtons();
        if (state.qualityStale && state.frameRanges !== null) await handlers.setFilter(null);
        updatePreviewButtons(); renderAnnotations(); handlers.drawChart();
    });
    $('stopQuality').onclick = run(async () => {
        if (jobId) { await api('/api/quality/stop', {id: jobId}); $('stopQuality').disabled = true; }
    });
    $('downloadReport').onclick = run(async () => {
        const saved = await api('/api/quality/save', {id: jobId});
        $('reportSavedPath').value = saved.path;
        $('qualityDetail').textContent = `已保存 ${saved.episodes} 条 Episode 的检测信息：\n${saved.path}\nmanifest.json：策略与统计；index.jsonl：轻量索引；episodes/：每条 Episode 的压缩 JSON 明细。`;
        $('reportSavedPath').focus(); $('reportSavedPath').select();
    });
    $('previewNormal').onclick = run(() => preview('normal'));
    $('previewSuspect').onclick = run(() => preview('suspect'));
    $('restoreAll').onclick = run(async () => { await handlers.setFilter(null); renderAnnotations(); handlers.drawChart(); });
    $('downloadSelection').onclick = () => {
        if (!report() || state.frameRanges === null) return;
        const mode = state.previewMode;
        const episodes = report().episodes.filter(row => state.frameRanges[row.episode_index]).map(row => ({
            episode_index: row.episode_index, ranges: mode === 'normal' ? row.normal_segments : row.suspect_segments,
        }));
        download(pretty({version: 1, kind: 'frame_segment_selection', source: report().source,
            policy: report().policy, mode, episodes, source_modified: false, dataset_exported: false}),
            'application/json', 'robotic_segment_selection.json');
    };
    for (const id of ['segmentType', 'segmentEpisode', 'includeWaiting']) $(id).onchange = () => { page = 0; renderSegments(); };
    $('segmentPrevious').onclick = () => { page--; renderSegments(); };
    $('segmentNext').onclick = () => { page++; renderSegments(); };
}
