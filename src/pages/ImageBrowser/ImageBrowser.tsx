import React, { useState, useEffect } from 'react';
import { Card, CardHeader, Button } from '../../components/common';

export const ImageBrowser: React.FC = () => {
  const [images, setImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 模拟加载图片列表
    setTimeout(() => {
      setImages([
        'https://via.placeholder.com/300x200?text=Image+1',
        'https://via.placeholder.com/300x200?text=Image+2',
        'https://via.placeholder.com/300x200?text=Image+3',
        'https://via.placeholder.com/300x200?text=Image+4',
        'https://via.placeholder.com/300x200?text=Image+5',
        'https://via.placeholder.com/300x200?text=Image+6',
      ]);
      setLoading(false);
    }, 500);
  }, []);

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>图像浏览器</h1>

      <div style={{ display: 'grid', gridTemplateColumns: selectedImage ? '1fr 300px' : '1fr', gap: 'var(--space-4)' }}>
        <Card>
          <CardHeader icon="🖼️" title={selectedImage ? '图片详情' : '图片列表'} />
          {loading ? (
            <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
          ) : selectedImage ? (
            <div>
              <img
                src={selectedImage}
                alt="Selected"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-4)' }}
              />
              <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                <Button variant="secondary" onClick={() => setSelectedImage(null)}>返回列表</Button>
                <Button variant="primary">下载</Button>
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 'var(--space-3)' }}>
              {images.map((img, idx) => (
                <div
                  key={idx}
                  onClick={() => setSelectedImage(img)}
                  style={{
                    cursor: 'pointer',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-md)',
                    overflow: 'hidden',
                    transition: 'transform 0.2s',
                  }}
                >
                  <img src={img} alt={`Image ${idx + 1}`} style={{ width: '100%', display: 'block' }} />
                  <div style={{ padding: 'var(--space-2)', fontSize: '0.75rem', textAlign: 'center' }}>
                    Image {idx + 1}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {selectedImage && (
          <Card>
            <CardHeader icon="ℹ️" title="图片信息" />
            <div style={{ fontSize: '0.875rem' }}>
              <p><strong>文件名:</strong> image_{selectedImage.match(/\d+/)?.[0] || 'unknown'}.jpg</p>
              <p><strong>尺寸:</strong> 1920 x 1080</p>
              <p><strong>大小:</strong> 2.5 MB</p>
              <p><strong>格式:</strong> JPEG</p>
              <p><strong>创建时间:</strong> 2024-01-15</p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};
