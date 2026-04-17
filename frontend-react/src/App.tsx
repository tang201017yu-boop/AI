import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout';
import { Home } from './pages/Home/Home';
import { Inference } from './pages/Inference/Inference';
import { Training } from './pages/Training/Training';
import { Models } from './pages/Models/Models';
import { Datasets } from './pages/Datasets/Datasets';
import { Annotation } from './pages/Annotation/Annotation';
import { Solutions } from './pages/Solutions/Solutions';
import { Augmentation } from './pages/Augmentation/Augmentation';
import { ImageBrowser } from './pages/ImageBrowser/ImageBrowser';
import { TrainingMonitor } from './pages/TrainingMonitor/TrainingMonitor';
import { TrainingDetails } from './pages/TrainingDetails/TrainingDetails';
import { Projects } from './pages/Projects/Projects';
import { AINative } from './pages/AINative/AINative';

export const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/embed/annotation" element={<Annotation />} />
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="ai-native" element={<AINative />} />
        <Route path="inference" element={<Inference />} />
        <Route path="training" element={<Training />} />
        <Route path="models" element={<Models />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="annotation" element={<Annotation />} />
        <Route path="solutions" element={<Solutions />} />
        <Route path="augmentation" element={<Augmentation />} />
        <Route path="image-browser" element={<ImageBrowser />} />
        <Route path="training-monitor" element={<TrainingMonitor />} />
        <Route path="training/:taskId/details" element={<TrainingDetails />} />
        <Route path="projects" element={<Projects />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
};

export default App;
