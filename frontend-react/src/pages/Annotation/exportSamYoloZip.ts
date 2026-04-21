import JSZip from 'jszip';
import type { SAMAnnotation } from '../../types';
import { buildYoloSingleImageExport, stemFromImageName } from './exportSamYolo';

export class YoloZipExportError extends Error {
  constructor(
    message: string,
    public readonly code: 'NO_ANNOTATIONS' | 'NO_SIZE' | 'NO_LINES' | 'NO_IMAGE' | 'ZIP_FAILED'
  ) {
    super(message);
    this.name = 'YoloZipExportError';
  }
}

/**
 * 打包标准 YOLO 单图目录：images/原图、labels/同名 stem.txt、classes.txt、README.txt
 */
export async function buildYoloSingleImageZipBlob(params: {
  annotations: SAMAnnotation[];
  samClasses: string[];
  imgW: number;
  imgH: number;
  /** 磁盘文件名，如 foo.jpg（决定 stem 与 labels/foo.txt） */
  imageBaseName: string;
  imageBytes: ArrayBuffer;
}): Promise<Blob> {
  const { annotations, samClasses, imgW, imgH, imageBaseName, imageBytes } = params;
  if (!annotations.length) {
    throw new YoloZipExportError('没有标注', 'NO_ANNOTATIONS');
  }
  if (!imgW || !imgH) {
    throw new YoloZipExportError('缺少图片尺寸', 'NO_SIZE');
  }
  const { lines, classNames } = buildYoloSingleImageExport(annotations, samClasses, imgW, imgH);
  if (!lines.length) {
    throw new YoloZipExportError('没有可写入 labels 的框或多边形', 'NO_LINES');
  }
  const stem = stemFromImageName(imageBaseName);
  const safeImageName = String(imageBaseName || `${stem}.jpg`)
    .replace(/^[/\\]+/, '')
    .replace(/\.\./g, '_')
    .replace(/[/\\]/g, '_')
    || `${stem}.jpg`;

  const zip = new JSZip();
  zip.folder('images')?.file(safeImageName, imageBytes);
  zip.folder('labels')?.file(`${stem}.txt`, lines.join('\n'));
  zip.file('classes.txt', classNames.join('\n'));
  zip.file(
    'README.txt',
    [
      'YOLO 单图导出（Vision Platform）',
      '',
      `images/${safeImageName}  — 原图`,
      `labels/${stem}.txt       — 标注（class_id 与 classes.txt 行号一致，从 0 起）`,
      'classes.txt              — 本图类别名，一行一个',
      '',
      '将本 ZIP 解压后，可把「images」「labels」并入你的数据集目录；',
      '训练时请用 Ultralytics data.yaml 指向对应 train/val 路径。',
    ].join('\n'),
  );

  try {
    return await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
  } catch {
    throw new YoloZipExportError('生成 ZIP 失败', 'ZIP_FAILED');
  }
}
