// 页面入口：任务导航、帧加载、同步播放及各功能模块的连接。
import {$, state, api, pretty, format, error, run, cell, button, tasks, taskIds, visibleIds,
    frameCount, visibleFrame, nextFrame, findings} from './common.js';
import {bindChart, drawChart, renderDimensions} from './chart.js';
import {bindStatistics, renderStats} from './statistics.js';
import {bindQuality, restoreQuality, renderAnnotations, selectSegment} from './quality.js';
import {bindDataset} from './dataset.js';

function renderNavigation() {
    const list = tasks(), ids = visibleIds();
    const frames = ids.reduce((sum, id) => sum + frameCount(id), 0);
    $('overview').replaceChildren();
    for (const [title, value] of [
        ['Task 任务', list.length], ['Episode 轨迹', ids.length],
        ['数据帧', frames.toLocaleString()], ['采样帧率', `${state.info.fps} FPS`],
    ]) {
        const element = document.createElement('div');
        element.className = 'metric';
        const label = document.createElement('span');
        label.textContent = title;
        const number = document.createElement('strong');
        number.textContent = value;
        element.append(label, number);
        $('overview').append(element);
    }
    $('taskCount').textContent = list.length;
    $('tasks').replaceChildren();
    const query = $('taskSearch').value.toLowerCase();
    for (const task of [{name: null, count: ids.length, total_frames: frames}, ...list]) {
        if (task.name && !task.name.toLowerCase().includes(query)) continue;
        const element = button(task.name || '全部任务', () => selectTask(task.name));
        const meta = document.createElement('small');
        meta.textContent = `${task.count} 条轨迹 · ${task.total_frames.toLocaleString()} 帧`;
        element.append(meta);
        element.classList.toggle('selected', task.name === state.task);
        $('tasks').append(element);
    }
    renderEpisodes();
    $('previewBanner').hidden = state.filter === null;
    $('previewLabel').textContent = state.previewMode === 'suspect'
        ? '正在预览疑似异常区间，帧号和时间戳保持原始编号'
        : '正在预览未标记异常的区间（仅针对已执行的检查），原始数据未改变';
}

function renderEpisodes() {
    const ids = taskIds();
    $('episodeCount').textContent = ids.length;
    $('episodes').replaceChildren();
    for (const id of ids) {
        const meta = state.info.episodes[id];
        const element = button(`Episode ${id}`, () => loadEpisode(id));
        element.dataset.episode = id;
        const sub = document.createElement('small');
        sub.textContent = state.frameRanges === null
            ? `${meta.length} 帧 · ${(meta.length / state.info.fps).toFixed(2)} s`
            : `预览 ${frameCount(id)} / ${meta.length} 帧`;
        element.append(sub);
        const badge = document.createElement('small');
        badge.className = 'episode-quality';
        const result = findings(id);
        badge.textContent = result ? `疑似异常 ${result.suspect_frames} 帧${result.episode_complete === false ? ' · 检测中' : ''}` : '';
        element.append(badge);
        element.classList.toggle('selected', state.episode?.index === id);
        $('episodes').append(element);
    }
    if (!ids.length) {
        const empty = document.createElement('p');
        empty.className = 'empty';
        empty.textContent = '当前范围没有轨迹';
        $('episodes').append(empty);
    }
}

function selectTask(name) {
    state.task = name;
    renderNavigation();
    const ids = taskIds();
    if (ids.length && !ids.includes(state.episode?.index)) return loadEpisode(ids[0]);
}

function stop() {
    state.playing = false;
    clearTimeout(state.timer);
    $('play').textContent = '▶ 播放';
}

function clearEpisode() {
    stop();
    state.generation++;
    state.request++;
    state.controller?.abort();
    state.episode = null;
    state.frame = 0;
    $('episodeTitle').textContent = '当前范围没有轨迹';
    $('episodeInfo').textContent = '可恢复全部数据或选择其他任务';
    for (const id of ['cameras', 'values', 'dimensions']) $(id).replaceChildren();
    $('rawValues').textContent = '';
    $('frameInfo').textContent = '';
    for (const id of ['frameSlider', 'frameNumber']) {
        $(id).value = 0;
        $(id).max = 0;
        $(id).disabled = true;
    }
    $('frameTotal').textContent = '/ 0';
    $('play').disabled = true;
    drawChart();
    renderAnnotations();
}

async function loadEpisode(id, frame = 0) {
    if (!state.info.episodes[id]) throw new Error(`找不到 Episode ${id}`);
    stop();
    const generation = ++state.generation;
    state.request++;
    state.controller?.abort();
    state.episode = null;
    for (const name of ['play', 'frameSlider', 'frameNumber']) $(name).disabled = true;
    $('episodeTitle').textContent = `正在加载 Episode ${id}…`;
    $('cameras').replaceChildren();
    $('values').replaceChildren();
    $('rawValues').textContent = '';
    drawChart();
    const episode = await api(`/api/episode?episode=${id}`);
    if (generation !== state.generation) return;
    state.episode = episode;
    state.zoom = null;
    state.dims = new Set(Array.from({
        length: Math.max(state.info.schema.state.names.length, state.info.schema.action.names.length),
    }, (_, i) => i));
    $('episodeTitle').textContent = `Episode ${id}`;
    $('episodeInfo').textContent = `${episode.length} 原始帧${state.frameRanges === null ? '' : ` · 预览 ${frameCount(id)} 帧`} · ${(episode.length / state.info.fps).toFixed(2)} s · ${
        state.info.fps} FPS · State: ${state.info.schema.state.key || '未映射'} / Action: ${state.info.schema.action.key || '未映射'}`;
    $('metadata').textContent = pretty({
        ...episode.metadata,
        timestamp_start: episode.times[0],
        timestamp_end: episode.times.at(-1),
        timestamp_span: episode.times.at(-1) != null && episode.times[0] != null
            ? episode.times.at(-1) - episode.times[0] : null,
        state: state.info.schema.state,
        action: state.info.schema.action,
    });
    for (const key of ['frameSlider', 'frameNumber']) {
        $(key).max = episode.length - 1;
        $(key).disabled = false;
    }
    $('frameTotal').textContent = `/ ${episode.length - 1}`;
    $('play').disabled = false;
    renderDimensions();
    renderEpisodes();
    renderAnnotations();
    await showFrame(Math.min(frame, episode.length - 1));
}

async function showFrame(frame) {
    const episode = state.episode;
    if (!episode) return;
    frame = Math.max(0, Math.min(episode.length - 1, Math.round(frame)));
    if (!visibleFrame(episode.index, frame)) {
        const direction = frame >= state.frame ? 1 : -1;
        frame = nextFrame(episode.index, frame, direction) ?? nextFrame(episode.index, frame, -direction);
        if (frame === null) return;
    }
    const request = ++state.request, generation = state.generation;
    state.controller?.abort();
    const controller = new AbortController();
    state.controller = controller;
    const data = await api(`/api/frame?episode=${episode.index}&frame=${frame}`, undefined, controller.signal);
    const decoded = await Promise.all(Object.entries(data.images).map(async ([camera, src]) => {
        const image = new Image();
        image.src = src;
        await image.decode();
        image.alt = `${camera} · Episode ${episode.index} · Frame ${frame}`;
        image.onclick = () => openImage(episode.index, frame, camera);
        const figure = document.createElement('figure');
        figure.className = 'camera';
        const caption = document.createElement('figcaption');
        caption.textContent = camera;
        figure.append(image, caption);
        return figure;
    }));
    // 多相机、数值和曲线在同一次提交中更新；过期响应不能覆盖新选择。
    if (request !== state.request || generation !== state.generation) return;
    state.frame = frame;
    $('episodeInfo').textContent = `${episode.length} 原始帧${state.frameRanges === null ? '' : ` · 预览 ${frameCount(episode.index)} 帧`} · ${(episode.length / state.info.fps).toFixed(2)} s · ${
        state.info.fps} FPS · State: ${state.info.schema.state.key || '未映射'} / Action: ${state.info.schema.action.key || '未映射'}`;
    $('frameSlider').value = frame;
    $('frameNumber').value = frame;
    $('cameras').replaceChildren(...decoded);
    for (const [camera, message] of Object.entries(data.errors)) {
        const figure = document.createElement('figure');
        figure.className = 'camera';
        const note = document.createElement('div');
        note.className = 'image-error';
        note.textContent = `${camera}：${message}`;
        figure.append(note);
        $('cameras').append(figure);
    }
    if (!state.info.cameras.length) {
        const element = document.createElement('p');
        element.className = 'empty';
        element.textContent = state.info.videos.length
            ? '当前数据包含外部视频，读取器暂不支持此存储格式' : '此数据集没有内嵌相机图像';
        $('cameras').append(element);
    }
    $('frameInfo').textContent = `Frame ${frame} · t = ${format(episode.times[frame])} s · ← / → 逐帧 · 空格播放 / 暂停`;
    $('rawValues').textContent = pretty(data.values);
    renderValues();
    drawChart();
    renderAnnotations();
}

function renderValues() {
    const episode = state.episode, frame = state.frame, schema = state.info.schema;
    $('values').replaceChildren();
    for (let i = 0; i < Math.max(schema.state.names.length, schema.action.names.length); i++) {
        const s = episode.state?.[frame]?.[i], a = episode.action?.[frame]?.[i];
        const row = document.createElement('tr');
        cell(row, i);
        cell(row, schema.state.names[i] || '—');
        cell(row, schema.action.names[i] || '—');
        cell(row, format(s));
        cell(row, format(a));
        cell(row, format(s == null || a == null ? null : a - s));
        $('values').append(row);
    }
}

async function tick() {
    if (!state.playing || !state.episode) return;
    const generation = state.generation, start = performance.now();
    let next = nextFrame(state.episode.index, state.frame + 1);
    if (next === null) {
        if ($('loop').checked) next = nextFrame(state.episode.index, 0);
        else { stop(); return; }
    }
    try {
        await showFrame(next);
    } catch (err) {
        if (err.name !== 'AbortError') { stop(); error(err); }
    }
    if (state.playing && generation === state.generation) {
        state.timer = setTimeout(tick, Math.max(0,
            1000 / (state.info.fps * Number($('speed').value)) - (performance.now() - start)));
    }
}

async function togglePlay() {
    if (!state.episode) return;
    if (state.playing) { stop(); return; }
    const generation = state.generation;
    if (nextFrame(state.episode.index, state.frame + 1) === null && !$('loop').checked) {
        await showFrame(0);
        if (generation !== state.generation) return;
    }
    state.playing = true;
    $('play').textContent = 'Ⅱ 暂停';
    state.timer = setTimeout(tick, 1000 / (state.info.fps * Number($('speed').value)));
}

function seek(frame) { stop(); return showFrame(frame); }

function moveEpisode(delta) {
    const ids = taskIds(), at = ids.indexOf(state.episode?.index), next = ids[at + delta];
    if (next !== undefined) return loadEpisode(next);
}

function openImage(episode, frame, camera) {
    $('imageTitle').textContent = `${camera} · Episode ${episode} · Frame ${frame}`;
    $('fullImage').src = `/api/image?episode=${episode}&frame=${frame}&camera=${encodeURIComponent(camera)}&dataset=${encodeURIComponent(state.info.dataset_id)}`;
    $('imageDialog').showModal();
}

function setFilter(ids) {
    stop();
    if (ids === null) {
        state.frameRanges = null;
        state.previewMode = null;
    }
    state.filter = ids === null ? null : new Set(ids);
    if (state.task !== null && !tasks().some(task => task.name === state.task)) state.task = null;
    renderNavigation();
    const available = taskIds();
    if (!available.length) clearEpisode();
    else if (!available.includes(state.episode?.index)) return loadEpisode(available[0]);
    else if (state.episode) return showFrame(state.frame);
}

function setFrameFilter(ranges, mode) {
    state.frameRanges = ranges;
    state.previewMode = mode;
    return setFilter(Object.keys(ranges).map(Number));
}

function bindNavigation() {
    $('taskSearch').oninput = renderNavigation;
    $('episodeJump').onsubmit = run(async event => {
        event.preventDefault();
        const id = Number($('episodeId').value);
        if (!state.info.episodes[id]) throw new Error(`找不到 Episode ${id}`);
        if (state.filter !== null && !state.filter.has(id)) throw new Error('该轨迹不在当前预览范围，请先恢复全部数据');
        state.task = null;
        renderNavigation();
        await loadEpisode(id);
    });
    $('previousEpisode').onclick = run(() => moveEpisode(-1));
    $('nextEpisode').onclick = run(() => moveEpisode(1));
    $('play').onclick = run(togglePlay);
    $('frameSlider').oninput = run(() => seek(Number($('frameSlider').value)));
    $('frameNumber').onchange = run(() => seek(Number($('frameNumber').value)));
    $('metadataButton').onclick = () => { if (state.episode) $('metadataDialog').showModal(); };
    for (const element of document.querySelectorAll('[data-close]')) {
        element.onclick = () => $(element.dataset.close).close();
    }
    $('error').onclick = () => { $('error').hidden = true; };
    $('resetPreview').onclick = run(() => setFilter(null));
    document.addEventListener('keydown', run(event => {
        if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName)
            || document.querySelector('dialog[open]')) return;
        if (event.code === 'Space') { event.preventDefault(); return togglePlay(); }
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
            event.preventDefault();
            return seek(state.frame + (event.key === 'ArrowLeft' ? -1 : 1));
        }
    }));
    new ResizeObserver(() => {
        drawChart();
        if ($('statsDialog').open) renderStats();
    }).observe($('chart'));
}

async function openOriginalEpisode(id, frame, endFrame = frame) {
    state.filter = null;
    state.frameRanges = null;
    state.previewMode = null;
    state.task = null;
    renderNavigation();
    await loadEpisode(id, frame);
    if (endFrame > frame && state.episode?.index === id) {
        state.zoom = [Math.max(0, frame - 5), Math.min(state.episode.length - 1, endFrame + 5)];
        drawChart();
    }
}

async function init() {
    state.info = await api('/api/info');
    $('datasetPath').textContent = state.info.path;
    bindDataset(stop);
    bindNavigation();
    bindChart(seek, selectSegment, renderAnnotations);
    bindStatistics(selectTask);
    bindQuality({openEpisode: openOriginalEpisode, setFilter, setFrameFilter, drawChart, stopPlayback: stop});
    renderNavigation();
    const initial = state.info.initial_episode ?? visibleIds()[0];
    if (initial !== undefined) await loadEpisode(initial);
    else clearEpisode();
    await restoreQuality();
    document.body.dataset.ready = 'true';
}
run(init)();
