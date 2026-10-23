// 数据路径选择与服务端目录浏览；成功换源后刷新页面，重建全部浏览和质量状态。
import {$, state, api, run, button} from './common.js';

export function bindDataset(stopPlayback) {
    let switching = false;
    let directory = null;
    let directoryRequest = 0;
    $('datasetInput').value = state.info.path;

    async function browse(path) {
        const request = ++directoryRequest;
        directory = null;
        $('directoryStatus').textContent = '正在读取目录…';
        $('chooseDirectory').disabled = true;
        $('directoryParent').disabled = true;
        $('directoryList').replaceChildren();
        try {
            const data = await api('/api/directories', {path});
            if (request !== directoryRequest) return;
            directory = data;
            $('directoryInput').value = data.path;
            $('directoryParent').disabled = data.parent === data.path;
            $('chooseDirectory').disabled = !data.is_dataset;
            $('directoryStatus').textContent = data.is_dataset
                ? `此目录包含 LeRobot 元数据 · ${data.directories.length} 个子目录`
                : `${data.directories.length} 个子目录 · 请选择包含 meta/info.json 的数据集根目录`;
            for (const child of data.directories) {
                const entry = button(`📁 ${child.name}`, () => browse(child.path));
                entry.title = child.path;
                $('directoryList').append(entry);
            }
            if (!data.directories.length) {
                const empty = document.createElement('p');
                empty.className = 'empty';
                empty.textContent = '此目录没有可浏览的子文件夹';
                $('directoryList').append(empty);
            }
        } catch (err) {
            if (request !== directoryRequest) return;
            $('directoryStatus').textContent = `读取失败：${err.message}`;
        }
    }

    $('datasetForm').onsubmit = run(async event => {
        event.preventDefault();
        if (switching) return;
        const path = $('datasetInput').value.trim();
        if (!path) throw new Error('请输入数据集根目录');
        switching = true;
        stopPlayback();
        state.controller?.abort();
        $('error').hidden = true;
        $('datasetStatus').textContent = '正在打开数据集…';
        document.querySelector('main').inert = true;
        for (const id of ['openDataset', 'browseDataset', 'datasetInput', 'statsButton', 'qualityButton']) {
            $(id).disabled = true;
        }
        try {
            await api('/api/dataset/open', {path});
            state.generation++;
            state.request++;
            window.location.reload();
        } catch (err) {
            $('datasetStatus').textContent = '打开失败，当前数据集保持不变';
            throw err;
        } finally {
            switching = false;
            document.querySelector('main').inert = false;
            for (const id of ['openDataset', 'browseDataset', 'datasetInput', 'statsButton', 'qualityButton']) {
                $(id).disabled = false;
            }
        }
    });
    $('browseDataset').onclick = run(async () => {
        $('directoryInput').value = $('datasetInput').value.trim() || state.info.path;
        $('directoryDialog').showModal();
        await browse($('directoryInput').value);
    });
    $('directoryForm').onsubmit = run(event => {
        event.preventDefault();
        return browse($('directoryInput').value);
    });
    $('directoryParent').onclick = run(() => directory && browse(directory.parent));
    $('chooseDirectory').onclick = () => {
        if (!directory?.is_dataset) return;
        $('datasetInput').value = directory.path;
        $('directoryDialog').close();
        $('datasetForm').requestSubmit();
    };
}
