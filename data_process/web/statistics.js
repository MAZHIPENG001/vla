// 当前浏览范围的 Task 表格、长度分布及 CSV 下载。
import {$, state, tasks, visibleIds, format, cell, run, download, canvasContext} from './common.js';

let sort = 'name';
let direction = 1;
let selectTask;

export function renderStats() {
    const rows = tasks().sort((a, b) => direction * (typeof a[sort] === 'string'
        ? a[sort].localeCompare(b[sort]) : a[sort] - b[sort]));
    $('statsRows').replaceChildren();
    for (const task of rows) {
        const row = document.createElement('tr');
        for (const key of ['name', 'count', 'total_frames', 'mean_frames', 'seconds']) cell(row, format(task[key]));
        row.onclick = run(() => { $('statsDialog').close(); return selectTask(task.name); });
        $('statsRows').append(row);
    }
    const {ctx, w, h} = canvasContext($('histogram'));
    const lengths = visibleIds().map(id => state.info.episodes[id].length);
    if (!lengths.length) return;
    const max = lengths.reduce((value, length) => Math.max(value, length), 0);
    const bins = Array(20).fill(0);
    for (const length of lengths) bins[Math.min(19, Math.floor(length / Math.max(1, max) * 20))]++;
    const peak = Math.max(...bins), bottom = h - 30;
    ctx.fillStyle = '#147d78';
    for (let i = 0; i < bins.length; i++) {
        const height = bins[i] / peak * (bottom - 22);
        ctx.fillRect(35 + i * (w - 50) / 20, bottom - height, (w - 50) / 20 - 3, height);
    }
    ctx.fillStyle = '#788698';
    ctx.font = '11px sans-serif';
    ctx.fillText(`轨迹长度分布 · ${lengths.length} 条`, 4, 12);
    ctx.fillText('0 帧', 35, h - 8);
    ctx.fillText(`${max} 帧`, w - 65, h - 8);
}

export function bindStatistics(onSelectTask) {
    selectTask = onSelectTask;
    $('statsButton').onclick = () => { $('statsDialog').showModal(); renderStats(); };
    for (const element of document.querySelectorAll('[data-sort]')) {
        element.onclick = () => {
            direction = sort === element.dataset.sort ? -direction : 1;
            sort = element.dataset.sort;
            renderStats();
        };
    }
    $('exportStats').onclick = () => {
        const csvCell = value => '"' + String(value).replaceAll('"', '""') + '"';
        const rows = [
            ['Task', 'Episodes', 'Frames', 'Mean frames', 'Seconds'],
            ...tasks().map(task => [task.name, task.count, task.total_frames, task.mean_frames, task.seconds]),
        ];
        download('\ufeff' + rows.map(row => row.map(csvCell).join(',')).join('\r\n'),
            'text/csv;charset=utf-8', 'task_statistics.csv');
    };
}
