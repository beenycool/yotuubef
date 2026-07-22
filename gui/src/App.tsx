import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { ProjectView } from './pages/ProjectView';
import { ScriptStudio } from './pages/ScriptStudio';
import { Settings } from './pages/Settings';

import { ToastProvider } from './components/Toast';

export const App: React.FC = () => {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/project/:name" element={<ProjectView />} />
            <Route path="/project/:name/script" element={<ScriptStudio />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </ToastProvider>
  );
};

export default App;
