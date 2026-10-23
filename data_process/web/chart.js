// 曲线绘制、维度选择、缩放及跳帧手势。
import {$, state, palette, format, run, canvasContext, findings, anomalyStyles, visibleFrame, frameRanges} from './common.js';

let geometry = null;
let drag = null;

function bounds() {
    return state.zoom || [0, Math.max(1, (state.episode?.length || 2) - 1)];
}

export function renderDimensions() {
    const schema = state.info.schema;
    $('dimensions').replaceChildren();
    for (let i = 0; i < Math.max(schema.state.names.length, schema.action.names.length); i++) {
        const label = document.createElement('label');
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = state.dims.has(i);
        input.onchange = () => {
            input.checked ? state.dims.add(i) : state.dims.delete(i);
            drawChart();
        };
        label.append(input, document.createTextNode(`${i} · ${schema.state.names[i] || schema.action.names[i]}`));
        label.style.color = palette[i % palette.length];
        $('dimensions').append(label);
    }
}

export function drawChart() {
    const {ctx, w, h} = canvasContext($('chart'));
    geometry = null;
    if (!w || !h || !state.episode) return;
    const episode = state.episode;
    const [start, end] = bounds();
    const left = 62, right = w - 15, top = 44, bottom = h - 30;
    let low = Infinity, high = -Infinity;
    for (const data of [episode.state, episode.action]) {
        if (!data) continue;
        for (let frame = Math.max(0, Math.floor(start)); frame <= Math.min(episode.length - 1, Math.ceil(end)); frame++) {
            for (const dimension of state.dims) {
                const value = data[frame]?.[dimension];
                if (value != null && Number.isFinite(value)) {
                    low = Math.min(low, value);
                    high = Math.max(high, value);
                }
            }
        }
    }
    if (!Number.isFinite(low)) { low = -1; high = 1; }
    const margin = (high - low || 1) * .08;
    low -= margin;
    high += margin;
    const x = frame => left + (frame - start) / (end - start) * (right - left);
    const y = value => bottom - (value - low) / (high - low) * (bottom - top);
    geometry = {left, right, top, bottom, start, end};
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, 0, right - left, bottom);
    ctx.clip();
    if (state.frameRanges !== null) {
        ctx.fillStyle = '#657388';
        ctx.globalAlpha = .12;
        let previous = start - .5;
        for (const [a, b] of frameRanges(episode.index)) {
            if (b < start || a > end) continue;
            const next = Math.max(start - .5, a - .5);
            if (next > previous) ctx.fillRect(x(previous), top, x(next) - x(previous), bottom - top);
            previous = Math.max(previous, b + .5);
        }
        if (previous < end + .5) ctx.fillRect(x(previous), top, x(end + .5) - x(previous), bottom - top);
    }
    for (const segment of visibleSegments()) {
        const color = anomalyStyles[segment.type].color;
        const a = x(segment.start_frame - .5), b = x(segment.end_frame + .5);
        const lane = Object.keys(anomalyStyles).indexOf(segment.type);
        ctx.fillStyle = color;
        ctx.globalAlpha = segment.status === 'waiting' ? .035 : .08;
        ctx.fillRect(a, top, Math.max(2, b - a), bottom - top);
        ctx.globalAlpha = segment.status === 'waiting' ? .3 : .85;
        ctx.fillRect(a, 5 + lane * 5, Math.max(3, b - a), 4);
        if (state.selectedSegment?.id === segment.id) {
            ctx.globalAlpha = .8;
            ctx.strokeStyle = color;
            ctx.strokeRect(a, top, Math.max(2, b - a), bottom - top);
        }
    }
    ctx.restore();
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#8894a1';
    ctx.strokeStyle = '#e9eef3';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const yy = top + (bottom - top) * i / 4;
        ctx.beginPath();
        ctx.moveTo(left, yy);
        ctx.lineTo(right, yy);
        ctx.stroke();
        ctx.fillText((high - (high - low) * i / 4).toFixed(3), 2, yy + 3);
        const frame = start + (end - start) * i / 4;
        ctx.fillText(String(Math.round(frame)), x(frame) - 5, h - 9);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.clip();
    for (const [role, data] of [['state', episode.state], ['action', episode.action]]) {
        if (!data) continue;
        ctx.setLineDash(role === 'action' ? [5, 4] : []);
        for (const dimension of state.dims) {
            ctx.strokeStyle = palette[dimension % palette.length];
            ctx.lineWidth = 1.3;
            ctx.beginPath();
            let pen = false;
            for (let frame = Math.max(0, Math.floor(start)); frame <= Math.min(episode.length - 1, Math.ceil(end)); frame++) {
                const value = data[frame]?.[dimension];
                if (value == null || !Number.isFinite(value) || !visibleFrame(episode.index, frame)) { pen = false; continue; }
                if (pen) ctx.lineTo(x(frame), y(value));
                else ctx.moveTo(x(frame), y(value));
                pen = true;
            }
            ctx.stroke();
        }
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = '#253e54';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(state.frame), top);
    ctx.lineTo(x(state.frame), bottom);
    ctx.stroke();
    ctx.restore();
}

function frameAt(event) {
    if (!geometry) return 0;
    const px = event.clientX - $('chart').getBoundingClientRect().left;
    return geometry.start + (px - geometry.left) / (geometry.right - geometry.left) * (geometry.end - geometry.start);
}

function visibleSegments() {
    return (findings(state.episode?.index)?.segments || []).filter(segment =>
        state.anomalyTypes.has(segment.type) && (segment.status !== 'waiting' || state.showWaiting));
}

export function bindChart(seek, onSelectSegment, onChange) {
    for (const [type, style] of Object.entries(anomalyStyles)) {
        const label = document.createElement('label');
        label.style.color = style.color;
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = true;
        input.dataset.anomalyType = type;
        input.onchange = () => { input.checked ? state.anomalyTypes.add(type) : state.anomalyTypes.delete(type); drawChart(); onChange(); };
        label.append(input, document.createTextNode(style.label));
        $('anomalyLegend').append(label);
    }
    $('showWaiting').onchange = () => { state.showWaiting = $('showWaiting').checked; drawChart(); onChange(); };
    $('allDimensions').onclick = () => {
        state.dims = new Set(Array.from({
            length: Math.max(state.info.schema.state.names.length, state.info.schema.action.names.length),
        }, (_, i) => i));
        renderDimensions();
        drawChart();
    };
    $('noDimensions').onclick = () => { state.dims.clear(); renderDimensions(); drawChart(); };
    $('resetZoom').onclick = () => { state.zoom = null; drawChart(); };
    $('chart').onpointerdown = event => {
        if (event.shiftKey && state.episode) {
            drag = {x: event.clientX, bounds: [...bounds()]};
            $('chart').setPointerCapture(event.pointerId);
        }
    };
    $('chart').onpointerup = run(event => {
        if (drag) { drag = null; return; }
        if (state.episode) {
            const frame = Math.round(frameAt(event));
            const segment = visibleSegments().find(s => s.start_frame <= frame && frame <= s.end_frame);
            if (segment) onSelectSegment(segment, false);
            return seek(frame);
        }
    });
    $('chart').onpointercancel = () => { drag = null; };
    $('chart').onpointermove = event => {
        if (!state.episode || !geometry) return;
        if (drag) {
            const span = drag.bounds[1] - drag.bounds[0];
            const delta = (event.clientX - drag.x) / (geometry.right - geometry.left) * span;
            const start = Math.max(0, Math.min(state.episode.length - 1 - span, drag.bounds[0] - delta));
            state.zoom = [start, start + span];
            drawChart();
            return;
        }
        const frame = Math.max(0, Math.min(state.episode.length - 1, Math.round(frameAt(event))));
        const lines = [`Frame ${frame} · ${format(state.episode.times[frame])} s`];
        for (const dimension of state.dims) {
            lines.push(`${dimension}: S ${format(state.episode.state?.[frame]?.[dimension])} / A ${format(state.episode.action?.[frame]?.[dimension])}`);
        }
        for (const segment of visibleSegments().filter(s => s.start_frame <= frame && frame <= s.end_frame)) {
            lines.push(`${anomalyStyles[segment.type].label} ${segment.status === 'waiting' ? '等待' : '疑似异常'} · ${segment.role || segment.check} · ${segment.start_frame}–${segment.end_frame}`);
        }
        $('chartTooltip').textContent = lines.join('\n');
        $('chartTooltip').hidden = false;
    };
    $('chart').onpointerleave = () => { $('chartTooltip').hidden = true; };
    $('chart').addEventListener('wheel', event => {
        if (!state.episode || state.episode.length < 3) return;
        event.preventDefault();
        const [start, end] = bounds();
        const center = Math.max(start, Math.min(end, frameAt(event)));
        const max = state.episode.length - 1;
        const span = Math.min(max, Math.max(2, (end - start) * (event.deltaY > 0 ? 1.25 : .8)));
        const ratio = (center - start) / (end - start);
        const next = Math.max(0, Math.min(max - span, center - span * ratio));
        state.zoom = [next, next + span];
        drawChart();
    }, {passive: false});
}
