// 质量策略表单、后台任务轮询、报告复核及筛选预览。
import {$, state, api, pretty, error, run, cell, button, download} from './common.js';

const labels = {
    passed: '通过限定检查', review: '待复核', rejected: '剔除候选',
    unverified: '未验证', disabled: '未启用',
};
const signalFields = [
    ['kind', '信号含义'], ['dimensions', '参与维度（空=全部）'],
    ['static_epsilon', '静止容差（支持逗号向量）'], ['max_static_ratio', '静止比例上限'],
    ['velocity_zscore', '速度 Z-score（可空）'], ['acceleration_zscore', '加速度 Z-score（可空）'],
    ['jerk_limit', 'Jerk 上限（空=未校准）'], ['max_bad_ratio', '异常比例上限'],
];
const imageFields = [
    ['inspect_images', '检查图像'], ['image_stride', '图像采样步长'],
    ['image_width', '指标计算宽度'], ['blur_variance_min', '清晰度方差下限（空=未校准）'],
    ['max_bad_image_ratio', '模糊比例上限'], ['uniform_std_max', '低纹理标准差上限'],
    ['saturation_fraction', '近黑 / 白像素比例'], ['repeated_run', '连续重复图像数'],
    ['frame_gap_factor', '采样缺口倍数'], ['sync_tolerance_seconds', '同步容差（秒）'],
    ['state_timestamp_key', '状态时间戳字段'],
];
let report = null;
let reportPolicy = null;
let jobId = null;
let openEpisode;
let setFilter;

function setting(parent, id, title, value, kind) {
    const label = document.createElement('label');
    label.textContent = title;
    const input = document.createElement(kind === 'kind' ? 'select' : 'input');
    input.id = id;
    if (kind === 'kind') {
        for (const name of ['position', 'command', 'velocity', 'acceleration']) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            input.append(option);
        }
    } else input.type = typeof value === 'boolean' ? 'checkbox' : 'text';
    if (typeof value === 'boolean') input.checked = value;
    else input.value = value == null ? '' : Array.isArray(value) ? value.join(',') : value;
    label.append(input);
    parent.append(label);
}

function buildSettings() {
    for (const role of ['state', 'action']) {
        const group = document.createElement('div');
        group.className = 'signal-group';
        const heading = document.createElement('h3');
        heading.textContent = `${role.toUpperCase()} · ${state.info.schema[role].key || '未映射'}`;
        group.append(heading);
        for (const [key, title] of signalFields) {
            setting(group, `${role}_${key}`, title, state.info.policy[role][key], key);
        }
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
            else if (['static_epsilon', 'jerk_limit'].includes(key) && value.includes(',')) {
                rule[key] = value.split(',').map(x => numberValue(x));
            } else rule[key] = numberValue(value, ['velocity_zscore', 'acceleration_zscore', 'jerk_limit'].includes(key));
        }
        result[role] = rule;
    }
    for (const [key] of imageFields) {
        result[key] = key === 'inspect_images' ? $(key).checked : key === 'state_timestamp_key'
            ? $(key).value : numberValue($(key).value, key === 'blur_variance_min');
    }
    result.camera_timestamp_keys = JSON.parse($('cameraClocks').value);
    return result;
}

async function startQuality(event) {
    event.preventDefault();
    if (jobId) return;
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
    // 同时禁用按钮，避免双击在请求返回前提交重复任务。
    $('runQuality').disabled = true;
    try {
        const job = await api('/api/quality/start', {policy: currentPolicy, episode_ids: ids});
        report = null;
        reportPolicy = JSON.stringify(currentPolicy);
        jobId = job.id;
    } catch (err) {
        $('runQuality').disabled = false;
        throw err;
    }
    $('stopQuality').disabled = false;
    $('downloadReport').disabled = true;
    $('applyPreview').disabled = true;
    $('qualityRows').replaceChildren();
    $('funnel').replaceChildren();
    $('qualityDetail').textContent = '正在检查…';
    $('qualityProgress').max = ids.length;
    $('qualityProgress').value = 0;
    pollQuality();
}

async function pollQuality() {
    if (!jobId) return;
    try {
        const data = await api(`/api/quality/status?id=${encodeURIComponent(jobId)}`);
        $('qualityProgress').value = data.completed;
        $('qualityStatus').textContent = `正在检查 ${data.completed}/${data.total}${
            data.episode === undefined ? '' : ` · 最近 Episode ${data.episode}`}，可停止`;
        if (data.status === 'running') { setTimeout(pollQuality, 500); return; }
        jobId = null;
        $('runQuality').disabled = false;
        $('stopQuality').disabled = true;
        if (data.status === 'error') throw new Error(data.error);
        report = data.report;
        renderReport();
    } catch (err) {
        jobId = null;
        $('runQuality').disabled = false;
        $('stopQuality').disabled = true;
        $('qualityStatus').textContent = `检查失败：${err.message}`;
        error(err);
    }
}

function renderReport() {
    const summary = report.summary;
    $('qualityStatus').textContent = `${report.complete ? '检查完成' : '已停止（部分结果）'}：${
        summary.checked_episodes}/${summary.total_requested} 条 · 通过限定检查 ${summary.passed} · 待复核 ${
        summary.review} · 剔除候选 ${summary.rejected} · 候选 ${summary.candidate_frames} 帧 / ${
        summary.candidate_hours.toFixed(4)} 小时`;
    $('downloadReport').disabled = false;
    // 检查运行中允许改表单；不完整或参数不一致时只能下载报告。
    try {
        $('applyPreview').disabled = !report.complete || JSON.stringify(policy()) !== reportPolicy;
    } catch (_) {
        $('applyPreview').disabled = true;
    }
    $('funnel').replaceChildren();
    for (const stage of summary.funnel) {
        const row = document.createElement('tr');
        for (const key of ['stage', 'input_episodes', 'removed_episodes', 'remaining_episodes', 'remaining_frames']) {
            cell(row, stage[key]);
        }
        $('funnel').append(row);
    }
    $('qualityRows').replaceChildren();
    for (const result of report.episodes) {
        const row = document.createElement('tr');
        cell(row, result.episode_index);
        cell(row, result.tasks.join('; '));
        cell(row, result.length);
        cell(row, labels[result.decision], result.decision);
        cell(row, [...result.rejected_reasons, ...result.review_reasons].join('; '));
        cell(row, '').append(button('查看首个异常帧', () => {
            $('qualityDialog').close();
            return openEpisode(result.episode_index, result.flagged_frames[0] || 0);
        }));
        row.onclick = () => {
            $('qualityDetail').textContent = result.checks.map(check =>
                `[${labels[check.status] || check.status}] ${check.name}：${check.message}`
            ).join('\n') + '\n\n' + pretty(result);
        };
        $('qualityRows').append(row);
    }
}

export function bindQuality(handlers) {
    openEpisode = handlers.openEpisode;
    setFilter = handlers.setFilter;
    buildSettings();
    $('qualityButton').onclick = () => { $('qualityDialog').showModal(); };
    $('qualityForm').onsubmit = run(startQuality);
    $('qualityForm').oninput = () => { $('applyPreview').disabled = true; };
    $('stopQuality').onclick = run(async () => {
        if (jobId) {
            await api('/api/quality/stop', {id: jobId});
            $('stopQuality').disabled = true;
        }
    });
    $('downloadReport').onclick = () => {
        if (report) download(pretty(report), 'application/json', 'robotic_quality_report.json');
    };
    $('applyPreview').onclick = run(() => {
        if (!report?.complete) throw new Error('请先完成检查');
        if (JSON.stringify(policy()) !== reportPolicy) throw new Error('参数已修改，请重新检查');
        const ids = report.episodes.filter(row => row.decision === 'passed'
            || ($('includeReview').checked && row.decision === 'review')).map(row => row.episode_index);
        $('qualityDialog').close();
        return setFilter(ids);
    });
    $('restoreAll').onclick = run(() => setFilter(null));
}
