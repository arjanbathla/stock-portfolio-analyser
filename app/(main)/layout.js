"use client";

import Sidebar from "@/components/sidebar";
import { SidebarProvider, useSidebar } from "@/components/sidebar-context";

function MainContent({ children }) {
  const { isCollapsed } = useSidebar();
  
  return (
    <div className="min-h-screen bg-gray-50 text-black pt-20">
      <Sidebar />
      <div className={`transition-all duration-300 px-6 pb-6 ${
        isCollapsed ? 'ml-16' : 'ml-64'
      }`}>
        {children}
      </div>
    </div>
  );
}

export default function MainLayout({ children }) {
  return (
    <SidebarProvider>
      <MainContent>{children}</MainContent>
    </SidebarProvider>
  );
} 