import { NavLink } from "react-router-dom";

export default function Header() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? "bg-indigo-600 text-white"
        : "text-gray-400 hover:text-white hover:bg-gray-800"
    }`;

  return (
    <header className="border-b border-gray-800 bg-gray-900">
      <div className="container mx-auto px-4 max-w-6xl flex items-center justify-between h-14">
        <span className="font-bold text-lg tracking-tight text-white">
          Infer<span className="text-indigo-400">Mesh</span>
        </span>
        <nav className="flex gap-1">
          <NavLink to="/" end className={linkClass}>
            Dashboard
          </NavLink>
          <NavLink to="/playground" className={linkClass}>
            Playground
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
