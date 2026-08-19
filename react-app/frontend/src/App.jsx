import { Route, Routes } from "react-router-dom";

import Nav from "./components/Nav.jsx";
import ChatPage from "./pages/ChatPage.jsx";
import DatasetPage from "./pages/DatasetPage.jsx";
import ModelPage from "./pages/ModelPage.jsx";
import WhatIfPage from "./pages/WhatIfPage.jsx";

export default function App() {
  return (
    <div className="shell">
      <Nav />
      <main className="content">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/whatif" element={<WhatIfPage />} />
        </Routes>
      </main>
    </div>
  );
}
