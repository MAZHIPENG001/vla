// 页面共享状态、数据范围及通用 DOM / HTTP 操作。
export const $ = id => document.getElementById(id);

export const state = {
    info: null,
    task: null,
    filter: null,
    episode: null,
    frame: 0,
    generation: 0,
    request: 0,
    controller: null,
    playing: false,
    timer: null,
    dims: new Set(),
    zoom: null,
    frameRanges: null,
    previewMode: null,
    qualityReport: null,
    qualityDetails: new Map(),
    qualityStale: false,
    selectedSegment: null,
    anomalyTypes: new Set(['static', 'velocity', 'acceleration', 'jerk', 'integrity', 'image', 'sync']),
    showWaiting: true,
};

export const anomalyStyles = {
    static: {label: '静止', color: '#ba850c'},
    velocity: {label: '速度', color: '#d74b55'},
    acceleration: {label: '加速度', color: '#8054bd'},
    jerk: {label: 'Jerk', color: '#df711f'},
    integrity: {label: '完整性', color: '#ad4054'},
    image: {label: '图像', color: '#1b9994'},
    sync: {label: '同步', color: '#3a78be'},
};

export function findings(id) {
    return state.qualityDetails.get(id)?.row || state.qualityReport?.episodes.find(row => row.episode_index === id);
}

export function frameRanges(id) {
    return state.frameRanges === null ? [[0, state.info.episodes[id].length - 1]] : (state.frameRanges[id] || []);
}

export function frameCount(id) {
    return frameRanges(id).reduce((sum, [start, end]) => sum + end - start + 1, 0);
}

export function visibleFrame(id, frame) {
    return frameRanges(id).some(([start, end]) => start <= frame && frame <= end);
}

// 使用原始行索引预览区间，绝不将两段数据重新编号或拼接成连续信号。
export function nextFrame(id, requested, direction = 1) {
    const ranges = frameRanges(id);
    if (direction >= 0) {
        for (const [start, end] of ranges) if (requested <= end) return Math.max(start, requested);
    } else {
        for (let i = ranges.length - 1; i >= 0; i--) {
            const [start, end] = ranges[i];
            if (requested >= start) return Math.min(end, requested);
        }
    }
    return null;
}

export const palette = [
    '#147d78', '#4878c4', '#d29238', '#af5f91', '#5b9c62', '#cb6152', '#6d65ba',
    '#469eac', '#947742', '#ad78b7', '#3f6e83', '#8e9b44', '#d37e9b', '#747f93',
];

export const pretty = value => JSON.stringify(value, null, 2);
export const format = value => {
    if (value == null) return '—';
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(5);
    return String(value);
};

export function error(err) {
    $('error').textContent = err.message || String(err);
    $('error').hidden = false;
}

export async function api(path, body, signal) {
    const response = await fetch(path, {
        method: body ? 'POST' : 'GET',
        headers: {
            ...(state.info?.dataset_id ? {'X-Data-Process-Dataset': state.info.dataset_id} : {}),
            ...(body ? {
                'Content-Type': 'application/json',
                'X-Data-Process-Token': document.querySelector('meta[name=api-token]').content,
            } : {}),
        },
        body: body ? JSON.stringify(body) : undefined,
        signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

// 同步调用事件函数，确保 preventDefault 在浏览器执行默认行为之前生效。
export const run = fn => (...args) => {
    try {
        return Promise.resolve(fn(...args)).catch(err => {
            if (err.name !== 'AbortError') error(err);
        });
    } catch (err) {
        error(err);
        return Promise.resolve();
    }
};

export function cell(row, text, className) {
    const element = document.createElement('td');
    element.textContent = text;
    if (className) element.className = className;
    row.append(element);
    return element;
}

export function button(text, onClick) {
    const element = document.createElement('button');
    element.type = 'button';
    element.textContent = text;
    element.onclick = run(onClick);
    return element;
}

export function visibleIds() {
    return state.info ? Object.keys(state.info.episodes).map(Number)
        .filter(id => state.filter === null || state.filter.has(id)).sort((a, b) => a - b) : [];
}

export function tasks() {
    return state.info.tasks.map(task => {
        const ids = task.episode_ids.filter(id => state.filter === null || state.filter.has(id));
        const frames = ids.reduce((sum, id) => sum + frameCount(id), 0);
        return {
            ...task,
            episode_ids: ids,
            total_frames: frames,
            count: ids.length,
            mean_frames: ids.length ? frames / ids.length : 0,
            seconds: frames / state.info.fps,
        };
    }).filter(task => task.count);
}

export function taskIds() {
    return state.task === null ? visibleIds() : (tasks().find(task => task.name === state.task)?.episode_ids || []);
}

export function download(content, type, name) {
    const url = URL.createObjectURL(new Blob([content], {type}));
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function canvasContext(canvas) {
    const box = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(box.width * ratio);
    canvas.height = Math.round(box.height * ratio);
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    return {ctx, w: box.width, h: box.height};
}
