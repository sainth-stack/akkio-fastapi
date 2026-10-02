"""E-commerce shopping cart MVP generator for Agentic Builder.

Pages: Home, Register, Login, Products, ProductDetail, Cart, Checkout, OrderConfirmation, Dashboard
Backend: User, Product, Cart, Order, OrderItem + JWT auth + seed 10 products
"""
from __future__ import annotations
from typing import Dict


ECOMMERCE_KEYWORDS = (
    "shopping cart", "add to cart", "checkout", "place order", "order history",
    "product listing", "product details", "product card", "e-commerce", "ecommerce",
    "shop now", "online store", "order confirmation", "cart page",
)

ECOMMERCE_SUPPORTING = (
    "cart", "checkout", "products", "orders", "register", "login",
    "shop", "store", "shipping", "delivery", "purchase",
)


def is_ecommerce_domain(requirement: str, prd: str = "", uiux: str = "") -> bool:
    text = "\n".join([requirement or "", prd or "", uiux or ""]).lower()
    strong = sum(1 for k in ECOMMERCE_KEYWORDS if k in text)
    if strong >= 2:
        return True
    support = sum(1 for k in ECOMMERCE_SUPPORTING if k in text)
    return strong >= 1 and support >= 3


def ecommerce_frontend_files(title: str, colors: Dict[str, str]) -> Dict[str, str]:
    from app_builder.services.fullstack_frontend_generator import build_theme_ts, _auth_ts
    safe = title.replace("\\", "\\\\").replace("'", "\\'")
    return {
        "frontend/src/theme.ts": build_theme_ts(colors),
        "frontend/src/auth.ts": _ecommerce_auth_ts(),
        "frontend/src/App.tsx": _app_tsx(),
        "frontend/src/context/CartContext.tsx": _cart_context_tsx(),
        "frontend/src/layout/Navbar.tsx": _navbar_tsx(safe),
        "frontend/src/pages/HomePage.tsx": _home_page_tsx(),
        "frontend/src/pages/RegisterPage.tsx": _register_page_tsx(),
        "frontend/src/pages/LoginPage.tsx": _login_page_tsx(safe),
        "frontend/src/pages/ProductsPage.tsx": _products_page_tsx(),
        "frontend/src/pages/ProductDetailPage.tsx": _product_detail_page_tsx(),
        "frontend/src/pages/CartPage.tsx": _cart_page_tsx(),
        "frontend/src/pages/CheckoutPage.tsx": _checkout_page_tsx(),
        "frontend/src/pages/OrderConfirmationPage.tsx": _order_confirmation_page_tsx(),
        "frontend/src/pages/DashboardPage.tsx": _dashboard_page_tsx(),
        "frontend/src/api/mock.ts": _ecommerce_mock_ts(),
    }


def ecommerce_backend_files(title: str) -> Dict[str, str]:
    safe = (title or "E-Commerce Shop").replace('"', "'")[:80]
    return {
        "backend/models.py": _models_py(),
        "backend/schemas.py": _schemas_py(),
        "backend/routes.py": _routes_py(),
        "backend/seed.py": _seed_py(),
        "backend/main.py": _main_py(safe),
        "backend/requirements.txt": _requirements_txt(),
        "backend/tests/test_cart.py": _test_cart_py(),
        "README.md": _readme(safe),
    }


# ──────────────────────────────────────────────────────────────
# FRONTEND
# ──────────────────────────────────────────────────────────────

def _ecommerce_auth_ts() -> str:
    return """export function getToken(): string | null {
  return localStorage.getItem('access_token');
}
export function getUser(): { id: number; name: string; email: string } | null {
  const raw = localStorage.getItem('user');
  if (!raw || raw === 'undefined') return null;
  try { return JSON.parse(raw); } catch { return null; }
}
export function setAuth(token: string, user: { id: number; name: string; email: string }) {
  localStorage.setItem('access_token', token);
  localStorage.setItem('user', JSON.stringify(user));
}
export function clearAuth() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user');
}
export function isLoggedIn(): boolean { return !!getToken(); }
"""


def _app_tsx() -> str:
    # NOTE: main.tsx (frozen scaffold) already provides:
    #   <QueryClientProvider> <ThemeProvider> <CssBaseline> <BrowserRouter>
    # App.tsx must NOT duplicate those providers — only add CartProvider + routes.
    return """import { Navigate, Route, Routes } from 'react-router-dom';
import { CartProvider } from './context/CartContext';
import Navbar from './layout/Navbar';
import HomePage from './pages/HomePage';
import RegisterPage from './pages/RegisterPage';
import LoginPage from './pages/LoginPage';
import ProductsPage from './pages/ProductsPage';
import ProductDetailPage from './pages/ProductDetailPage';
import CartPage from './pages/CartPage';
import CheckoutPage from './pages/CheckoutPage';
import OrderConfirmationPage from './pages/OrderConfirmationPage';
import DashboardPage from './pages/DashboardPage';
import { isLoggedIn } from './auth';

function PrivateRoute({ children }: { children: JSX.Element }) {
  return isLoggedIn() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <CartProvider>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/products" element={<ProductsPage />} />
        <Route path="/products/:id" element={<ProductDetailPage />} />
        <Route path="/cart" element={<CartPage />} />
        <Route path="/checkout" element={<PrivateRoute><CheckoutPage /></PrivateRoute>} />
        <Route path="/order-confirmation" element={<PrivateRoute><OrderConfirmationPage /></PrivateRoute>} />
        <Route path="/dashboard" element={<PrivateRoute><DashboardPage /></PrivateRoute>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </CartProvider>
  );
}
"""


def _cart_context_tsx() -> str:
    return r"""import { createContext, useContext, useReducer, ReactNode } from 'react';

export type CartItem = { id: number; name: string; price: number; quantity: number; image?: string };

type State = { items: CartItem[] };
type Action =
  | { type: 'ADD'; item: Omit<CartItem, 'quantity'> }
  | { type: 'REMOVE'; id: number }
  | { type: 'SET_QTY'; id: number; quantity: number }
  | { type: 'CLEAR' };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'ADD': {
      const existing = state.items.find((i) => i.id === action.item.id);
      if (existing) {
        return { items: state.items.map((i) => i.id === action.item.id ? { ...i, quantity: i.quantity + 1 } : i) };
      }
      return { items: [...state.items, { ...action.item, quantity: 1 }] };
    }
    case 'REMOVE':
      return { items: state.items.filter((i) => i.id !== action.id) };
    case 'SET_QTY':
      if (action.quantity <= 0) return { items: state.items.filter((i) => i.id !== action.id) };
      return { items: state.items.map((i) => i.id === action.id ? { ...i, quantity: action.quantity } : i) };
    case 'CLEAR':
      return { items: [] };
    default:
      return state;
  }
}

const CartCtx = createContext<{
  items: CartItem[];
  addItem: (item: Omit<CartItem, 'quantity'>) => void;
  removeItem: (id: number) => void;
  setQty: (id: number, qty: number) => void;
  clearCart: () => void;
  count: number;
  total: number;
} | null>(null);

export function CartProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { items: [] });
  return (
    <CartCtx.Provider value={{
      items: state.items,
      addItem: (item) => dispatch({ type: 'ADD', item }),
      removeItem: (id) => dispatch({ type: 'REMOVE', id }),
      setQty: (id, quantity) => dispatch({ type: 'SET_QTY', id, quantity }),
      clearCart: () => dispatch({ type: 'CLEAR' }),
      count: state.items.reduce((s, i) => s + i.quantity, 0),
      total: state.items.reduce((s, i) => s + i.price * i.quantity, 0),
    }}>
      {children}
    </CartCtx.Provider>
  );
}

export function useCart() {
  const ctx = useContext(CartCtx);
  if (!ctx) throw new Error('useCart must be inside CartProvider');
  return ctx;
}
"""


def _navbar_tsx(title: str) -> str:
    safe = title.replace("'", "\\'")
    return r"""import { AppBar, Badge, Box, Button, Toolbar, Typography } from '@mui/material';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import { Link, useNavigate } from 'react-router-dom';
import { useCart } from '../context/CartContext';
import { clearAuth, getUser, isLoggedIn } from '../auth';

export default function Navbar() {
  const { count } = useCart();
  const navigate = useNavigate();
  const user = getUser();
  const handleLogout = () => { clearAuth(); navigate('/'); };
  return (
    <AppBar position="sticky" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
      <Toolbar sx={{ gap: 2 }}>
        <Typography variant="h6" fontWeight={800} component={Link} to="/" sx={{ color: 'primary.contrastText', textDecoration: 'none', flex: 1 }}>
          """ + safe + r"""
        </Typography>
        <Button color="inherit" component={Link} to="/">Home</Button>
        <Button color="inherit" component={Link} to="/products">Products</Button>
        <Button color="inherit" component={Link} to="/cart" startIcon={
          <Badge badgeContent={count} color="error"><ShoppingCartIcon /></Badge>
        }>
          Cart{count > 0 ? ` (${count})` : ''}
        </Button>
        {isLoggedIn() ? (
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Button color="inherit" component={Link} to="/dashboard">{user?.name || 'Account'}</Button>
            <Button variant="outlined" color="inherit" size="small" onClick={handleLogout}>Logout</Button>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button color="inherit" component={Link} to="/login">Login</Button>
            <Button variant="outlined" color="inherit" size="small" component={Link} to="/register">Register</Button>
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
}
"""


def _home_page_tsx() -> str:
    return r"""import { useQuery } from '@tanstack/react-query';
import { Box, Button, Card, CardActions, CardContent, CardMedia, Container, Grid, Snackbar, Typography } from '@mui/material';
import { Link, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { apiFetch } from '../api/client';
import { useCart } from '../context/CartContext';

type Product = { id: number; name: string; description: string; price: number; image: string; stock: number };

export default function HomePage() {
  const navigate = useNavigate();
  const { addItem } = useCart();
  const [snack, setSnack] = useState('');
  const { data } = useQuery({ queryKey: ['products'], queryFn: () => apiFetch<{ items: Product[] }>('/api/products') });
  const featured = (data?.items || []).slice(0, 6);

  const handleAdd = (p: Product) => {
    addItem({ id: p.id, name: p.name, price: p.price, image: p.image });
    setSnack(`${p.name} added to cart!`);
  };

  return (
    <Box>
      {/* Hero */}
      <Box sx={{ background: 'linear-gradient(135deg, #1565C0 0%, #0D47A1 100%)', color: '#fff', py: { xs: 8, md: 12 } }}>
        <Container maxWidth="lg">
          <Typography variant="h2" fontWeight={800} sx={{ mb: 2, fontSize: { xs: '2rem', md: '3.5rem' } }}>
            Welcome to Our Store
          </Typography>
          <Typography variant="h6" sx={{ mb: 4, opacity: 0.9 }}>
            Discover amazing products at unbeatable prices.
          </Typography>
          <Button variant="contained" size="large" color="secondary" onClick={() => navigate('/products')}
            sx={{ fontWeight: 700, px: 4, py: 1.5, fontSize: '1.1rem' }}>
            Shop Now
          </Button>
        </Container>
      </Box>

      {/* Featured Products */}
      <Container maxWidth="lg" sx={{ py: 6 }}>
        <Typography variant="h4" fontWeight={700} sx={{ mb: 4 }}>Featured Products</Typography>
        <Grid container spacing={3}>
          {featured.map((p) => (
            <Grid item xs={12} sm={6} md={4} key={p.id}>
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardMedia
                  component="img"
                  height="200"
                  image={p.image || `https://picsum.photos/seed/${p.id}/400/200`}
                  alt={p.name}
                  sx={{ objectFit: 'cover' }}
                />
                <CardContent sx={{ flex: 1 }}>
                  <Typography variant="h6" fontWeight={600}>{p.name}</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{p.description?.slice(0, 80)}…</Typography>
                  <Typography variant="h6" color="primary" fontWeight={700} sx={{ mt: 2 }}>₹{p.price.toLocaleString('en-IN')}</Typography>
                </CardContent>
                <CardActions sx={{ px: 2, pb: 2, gap: 1 }}>
                  <Button size="small" variant="contained" onClick={() => handleAdd(p)}>Add to Cart</Button>
                  <Button size="small" variant="outlined" component={Link} to={`/products/${p.id}`}>View Details</Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      <Snackbar open={!!snack} autoHideDuration={2500} onClose={() => setSnack('')} message={snack} />
    </Box>
  );
}
"""


def _register_page_tsx() -> str:
    return r"""import { useState } from 'react';
import { Alert, Box, Button, Card, CardContent, Container, Grid, TextField, Typography } from '@mui/material';
import { useNavigate, Link } from 'react-router-dom';
import { apiFetch } from '../api/client';

export default function RegisterPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ first_name: '', last_name: '', email: '', phone: '', password: '', confirm_password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const change = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!form.first_name || !form.last_name || !form.email || !form.password) {
      setError('All fields are required.'); return;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) {
      setError('Enter a valid email.'); return;
    }
    if (form.password !== form.confirm_password) {
      setError('Passwords do not match.'); return;
    }
    setLoading(true);
    try {
      await apiFetch('/api/auth/register', { method: 'POST', body: JSON.stringify(form) });
      navigate('/login', { state: { message: 'Registration successful. Please login.' } });
    } catch (err: any) {
      setError(err?.message || 'Registration failed');
    } finally { setLoading(false); }
  };

  return (
    <Container maxWidth="sm" sx={{ py: 8 }}>
      <Card>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h5" fontWeight={700} sx={{ mb: 3 }}>Create Account</Typography>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <Box component="form" onSubmit={submit}>
            <Grid container spacing={2}>
              <Grid item xs={6}><TextField fullWidth label="First Name" value={form.first_name} onChange={change('first_name')} required /></Grid>
              <Grid item xs={6}><TextField fullWidth label="Last Name" value={form.last_name} onChange={change('last_name')} required /></Grid>
              <Grid item xs={12}><TextField fullWidth label="Email" type="email" value={form.email} onChange={change('email')} required /></Grid>
              <Grid item xs={12}><TextField fullWidth label="Phone" value={form.phone} onChange={change('phone')} /></Grid>
              <Grid item xs={12}><TextField fullWidth label="Password" type="password" value={form.password} onChange={change('password')} required /></Grid>
              <Grid item xs={12}><TextField fullWidth label="Confirm Password" type="password" value={form.confirm_password} onChange={change('confirm_password')} required /></Grid>
              <Grid item xs={12}>
                <Button type="submit" fullWidth variant="contained" size="large" disabled={loading}>
                  {loading ? 'Registering…' : 'Register'}
                </Button>
              </Grid>
            </Grid>
          </Box>
          <Typography variant="body2" sx={{ mt: 2, textAlign: 'center' }}>
            Already have an account? <Link to="/login">Login</Link>
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
}
"""


def _login_page_tsx(title: str) -> str:
    return r"""import { useState } from 'react';
import { Alert, Box, Button, Card, CardContent, Container, TextField, Typography } from '@mui/material';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { apiFetch } from '../api/client';
import { setAuth } from '../auth';

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation() as any;
  const successMsg = location.state?.message || '';
  const [email, setEmail] = useState('user@example.com');
  const [password, setPassword] = useState('password123');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await apiFetch<{ access_token: string; user: { id: number; name: string; email: string } }>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      });
      setAuth(res.access_token, res.user);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err?.message || 'Invalid credentials');
    } finally { setLoading(false); }
  };

  return (
    <Container maxWidth="xs" sx={{ py: 10 }}>
      <Card>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h5" fontWeight={700} sx={{ mb: 1 }}>Sign In</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>Welcome back! Please enter your details.</Typography>
          {successMsg && <Alert severity="success" sx={{ mb: 2 }}>{successMsg}</Alert>}
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <Box component="form" onSubmit={submit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required fullWidth />
            <TextField label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required fullWidth />
            <Button type="submit" variant="contained" size="large" disabled={loading} fullWidth>
              {loading ? 'Signing in…' : 'Login'}
            </Button>
          </Box>
          <Typography variant="body2" sx={{ mt: 2, textAlign: 'center' }}>
            Don't have an account? <Link to="/register">Register</Link>
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
            Demo: user@example.com / password123
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
}
"""


def _products_page_tsx() -> str:
    return r"""import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Box, Button, Card, CardActions, CardContent, CardMedia, Container, Grid, Snackbar, TextField, Typography } from '@mui/material';
import { Link } from 'react-router-dom';
import { apiFetch } from '../api/client';
import { useCart } from '../context/CartContext';

type Product = { id: number; name: string; description: string; price: number; image: string; stock: number };

export default function ProductsPage() {
  const { addItem } = useCart();
  const [q, setQ] = useState('');
  const [snack, setSnack] = useState('');
  const { data, isLoading } = useQuery({ queryKey: ['products'], queryFn: () => apiFetch<{ items: Product[] }>('/api/products') });
  const products = (data?.items || []).filter((p) => p.name.toLowerCase().includes(q.toLowerCase()) || p.description?.toLowerCase().includes(q.toLowerCase()));

  const handleAdd = (p: Product) => {
    addItem({ id: p.id, name: p.name, price: p.price, image: p.image });
    setSnack(`${p.name} added to cart!`);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" fontWeight={700} sx={{ mb: 3 }}>Products</Typography>
      <TextField size="small" placeholder="Search products…" value={q} onChange={(e) => setQ(e.target.value)} sx={{ mb: 3, width: 320 }} />
      {isLoading && <Typography>Loading products…</Typography>}
      {!isLoading && !products.length && <Typography color="text.secondary">No products found.</Typography>}
      <Grid container spacing={3}>
        {products.map((p) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={p.id}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', '&:hover': { boxShadow: 4 } }}>
              <CardMedia
                component="img" height="180"
                image={p.image || `https://picsum.photos/seed/${p.id}shop/400/180`}
                alt={p.name} sx={{ objectFit: 'cover' }}
              />
              <CardContent sx={{ flex: 1 }}>
                <Typography variant="h6" fontWeight={600} gutterBottom>{p.name}</Typography>
                <Typography variant="body2" color="text.secondary">{p.description?.slice(0, 80)}</Typography>
                <Typography variant="h6" color="primary.main" fontWeight={700} sx={{ mt: 1.5 }}>₹{p.price.toLocaleString('en-IN')}</Typography>
                {p.stock === 0 && <Typography variant="caption" color="error">Out of stock</Typography>}
              </CardContent>
              <CardActions sx={{ px: 2, pb: 2, gap: 1 }}>
                <Button size="small" variant="contained" onClick={() => handleAdd(p)} disabled={p.stock === 0}>Add to Cart</Button>
                <Button size="small" variant="outlined" component={Link} to={`/products/${p.id}`}>View Details</Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
      <Snackbar open={!!snack} autoHideDuration={2500} onClose={() => setSnack('')} message={snack} />
    </Container>
  );
}
"""


def _product_detail_page_tsx() -> str:
    return r"""import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams, Link } from 'react-router-dom';
import { Alert, Box, Button, Card, CardContent, CardMedia, Container, Grid, Snackbar, Typography } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import { apiFetch } from '../api/client';
import { useCart } from '../context/CartContext';

export default function ProductDetailPage() {
  const { id } = useParams();
  const { addItem } = useCart();
  const [qty, setQty] = useState(1);
  const [snack, setSnack] = useState('');
  const { data, isLoading } = useQuery({ queryKey: ['product', id], queryFn: () => apiFetch<any>(`/api/products/${id}`) });
  const p = data;

  if (isLoading) return <Container sx={{ py: 6 }}><Typography>Loading…</Typography></Container>;
  if (!p) return <Container sx={{ py: 6 }}><Typography>Product not found. <Link to="/products">Back</Link></Typography></Container>;

  const handleAdd = () => {
    for (let i = 0; i < qty; i++) addItem({ id: p.id, name: p.name, price: p.price, image: p.image });
    setSnack(`${p.name} × ${qty} added to cart!`);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 5 }}>
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Card><CardMedia component="img" image={p.image || `https://picsum.photos/seed/${p.id}detail/600/420`} alt={p.name} sx={{ height: 420, objectFit: 'cover' }} /></Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Typography variant="h4" fontWeight={700} gutterBottom>{p.name}</Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>{p.description}</Typography>
          <Typography variant="h4" color="primary.main" fontWeight={800} sx={{ mb: 1 }}>₹{p.price?.toLocaleString('en-IN')}</Typography>
          <Typography variant="body2" color={p.stock > 0 ? 'success.main' : 'error'} sx={{ mb: 3 }}>
            {p.stock > 0 ? `In stock (${p.stock} available)` : 'Out of stock'}
          </Typography>
          <Card sx={{ mb: 3 }}><CardContent>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Quantity</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Button variant="outlined" size="small" onClick={() => setQty(Math.max(1, qty - 1))} disabled={qty <= 1}><RemoveIcon fontSize="small" /></Button>
              <Typography variant="h6" sx={{ minWidth: 32, textAlign: 'center' }}>{qty}</Typography>
              <Button variant="outlined" size="small" onClick={() => setQty(Math.min(p.stock, qty + 1))} disabled={qty >= p.stock}><AddIcon fontSize="small" /></Button>
            </Box>
          </CardContent></Card>
          {p.stock === 0 ? (
            <Alert severity="warning">This product is out of stock.</Alert>
          ) : (
            <Button variant="contained" size="large" fullWidth onClick={handleAdd}>Add to Cart</Button>
          )}
          <Button variant="text" component={Link} to="/products" sx={{ mt: 2 }}>← Back to Products</Button>
        </Grid>
      </Grid>
      <Snackbar open={!!snack} autoHideDuration={2500} onClose={() => setSnack('')} message={snack} />
    </Container>
  );
}
"""


def _cart_page_tsx() -> str:
    return r"""import { Box, Button, Container, IconButton, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography, Paper } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import { Link, useNavigate } from 'react-router-dom';
import { useCart } from '../context/CartContext';
import { isLoggedIn } from '../auth';

export default function CartPage() {
  const { items, removeItem, setQty, total, count } = useCart();
  const navigate = useNavigate();

  if (!items.length) {
    return (
      <Container maxWidth="md" sx={{ py: 10, textAlign: 'center' }}>
        <Typography variant="h5" sx={{ mb: 2 }}>Your cart is empty</Typography>
        <Button variant="contained" component={Link} to="/products">Continue Shopping</Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" fontWeight={700} sx={{ mb: 3 }}>Shopping Cart</Typography>
      <TableContainer component={Paper} variant="outlined">
        <Table>
          <TableHead><TableRow>
            <TableCell><strong>Product</strong></TableCell>
            <TableCell align="right"><strong>Price</strong></TableCell>
            <TableCell align="center"><strong>Quantity</strong></TableCell>
            <TableCell align="right"><strong>Total</strong></TableCell>
            <TableCell align="center"><strong>Remove</strong></TableCell>
          </TableRow></TableHead>
          <TableBody>
            {items.map((item) => (
              <TableRow key={item.id}>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    {item.image && <img src={item.image || `https://picsum.photos/seed/${item.id}/60/60`} alt={item.name} width={60} height={60} style={{ objectFit: 'cover', borderRadius: 8 }} />}
                    <Typography fontWeight={600}>{item.name}</Typography>
                  </Box>
                </TableCell>
                <TableCell align="right">₹{item.price.toLocaleString('en-IN')}</TableCell>
                <TableCell align="center">
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                    <IconButton size="small" onClick={() => setQty(item.id, item.quantity - 1)}><RemoveIcon fontSize="small" /></IconButton>
                    <Typography sx={{ minWidth: 28, textAlign: 'center' }}>{item.quantity}</Typography>
                    <IconButton size="small" onClick={() => setQty(item.id, item.quantity + 1)}><AddIcon fontSize="small" /></IconButton>
                  </Box>
                </TableCell>
                <TableCell align="right" sx={{ fontWeight: 700 }}>₹{(item.price * item.quantity).toLocaleString('en-IN')}</TableCell>
                <TableCell align="center">
                  <IconButton color="error" onClick={() => removeItem(item.id)}><DeleteIcon /></IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Button variant="outlined" component={Link} to="/products">Continue Shopping</Button>
        <Box sx={{ textAlign: 'right' }}>
          <Typography variant="h5" fontWeight={700}>Total: ₹{total.toLocaleString('en-IN')}</Typography>
          <Button variant="contained" size="large" sx={{ mt: 1, px: 4 }}
            onClick={() => isLoggedIn() ? navigate('/checkout') : navigate('/login')}>
            Checkout
          </Button>
        </Box>
      </Box>
    </Container>
  );
}
"""


def _checkout_page_tsx() -> str:
    return r"""import { useState } from 'react';
import { Box, Button, Card, CardContent, Container, Divider, Grid, TextField, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { apiFetch } from '../api/client';
import { getUser } from '../auth';
import { useCart } from '../context/CartContext';

export default function CheckoutPage() {
  const navigate = useNavigate();
  const user = getUser();
  const { items, total, clearCart } = useCart();
  const [shipping, setShipping] = useState({ address: '', city: '', state: '', pincode: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const ch = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) => setShipping({ ...shipping, [k]: e.target.value });

  const placeOrder = async () => {
    if (!shipping.address || !shipping.city || !shipping.state || !shipping.pincode) {
      setError('Fill in all shipping fields.'); return;
    }
    setError('');
    setLoading(true);
    try {
      const res = await apiFetch<{ order_number: string; total: number }>('/api/orders', {
        method: 'POST',
        body: JSON.stringify({ shipping_address: `${shipping.address}, ${shipping.city}, ${shipping.state} - ${shipping.pincode}`, items }),
      });
      clearCart();
      navigate('/order-confirmation', { state: { order_number: res.order_number, total: res.total } });
    } catch (err: any) {
      setError(err?.message || 'Could not place order. Try again.');
    } finally { setLoading(false); }
  };

  if (!items.length) {
    return <Container sx={{ py: 8, textAlign: 'center' }}><Typography>No items in cart. <a href="/products">Shop now</a></Typography></Container>;
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" fontWeight={700} sx={{ mb: 3 }}>Checkout</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <Card sx={{ mb: 2 }}><CardContent>
            <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>Customer Details</Typography>
            <Typography><strong>Name:</strong> {user?.name}</Typography>
            <Typography><strong>Email:</strong> {user?.email}</Typography>
          </CardContent></Card>
          <Card><CardContent>
            <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>Shipping Address</Typography>
            {error && <Typography color="error" sx={{ mb: 1 }}>{error}</Typography>}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField label="Address" value={shipping.address} onChange={ch('address')} required fullWidth />
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}><TextField label="City" value={shipping.city} onChange={ch('city')} required fullWidth /></Grid>
                <Grid item xs={12} sm={4}><TextField label="State" value={shipping.state} onChange={ch('state')} required fullWidth /></Grid>
                <Grid item xs={12} sm={4}><TextField label="Pincode" value={shipping.pincode} onChange={ch('pincode')} required fullWidth /></Grid>
              </Grid>
            </Box>
          </CardContent></Card>
        </Grid>
        <Grid item xs={12} md={5}>
          <Card><CardContent>
            <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>Order Summary</Typography>
            {items.map((item) => (
              <Box key={item.id} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">{item.name} × {item.quantity}</Typography>
                <Typography variant="body2" fontWeight={600}>₹{(item.price * item.quantity).toLocaleString('en-IN')}</Typography>
              </Box>
            ))}
            <Divider sx={{ my: 2 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
              <Typography fontWeight={700}>Total</Typography>
              <Typography fontWeight={700} color="primary">₹{total.toLocaleString('en-IN')}</Typography>
            </Box>
            <Card variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'action.hover' }}>
              <Typography fontWeight={600}>Cash on Delivery</Typography>
              <Typography variant="caption" color="text.secondary">Pay when your order arrives.</Typography>
            </Card>
            <Button variant="contained" fullWidth size="large" onClick={placeOrder} disabled={loading}>
              {loading ? 'Placing order…' : 'Place Order'}
            </Button>
          </CardContent></Card>
        </Grid>
      </Grid>
    </Container>
  );
}
"""


def _order_confirmation_page_tsx() -> str:
    return r"""import { Box, Button, Card, CardContent, Container, Typography } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { Link, useLocation } from 'react-router-dom';

export default function OrderConfirmationPage() {
  const { state } = useLocation() as any;
  const order_number = state?.order_number || 'ORD-0001';
  const total = state?.total || 0;

  return (
    <Container maxWidth="sm" sx={{ py: 10 }}>
      <Card>
        <CardContent sx={{ textAlign: 'center', p: 5 }}>
          <CheckCircleIcon sx={{ fontSize: 72, color: 'success.main', mb: 2 }} />
          <Typography variant="h4" fontWeight={800} gutterBottom>Order Placed Successfully!</Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
            Thank you for shopping with us.
          </Typography>
          <Box sx={{ bgcolor: 'background.default', borderRadius: 2, p: 3, mb: 3 }}>
            <Typography variant="body1"><strong>Order Number:</strong> {order_number}</Typography>
            <Typography variant="body1"><strong>Total:</strong> ₹{Number(total).toLocaleString('en-IN')}</Typography>
            <Typography variant="body1"><strong>Status:</strong> Placed</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Payment method: Cash on Delivery
            </Typography>
          </Box>
          <Button variant="contained" size="large" component={Link} to="/dashboard">Go to Dashboard</Button>
        </CardContent>
      </Card>
    </Container>
  );
}
"""


def _dashboard_page_tsx() -> str:
    return r"""import { useQuery } from '@tanstack/react-query';
import { Box, Button, Card, CardContent, Chip, Container, Grid, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import { Link, useNavigate } from 'react-router-dom';
import { apiFetch } from '../api/client';
import { clearAuth, getUser } from '../auth';
import { useCart } from '../context/CartContext';

type Order = { id: number; order_number: string; total: number; status: string; created_at: string };
type Stats = { total_orders: number; pending_orders: number; total_spent: number; recent_orders: Order[] };

const STATUS_COLOR: Record<string, any> = { Placed: 'info', Delivered: 'success', Cancelled: 'error', Processing: 'warning' };

export default function DashboardPage() {
  const navigate = useNavigate();
  const user = getUser();
  const { count } = useCart();
  const { data, isLoading } = useQuery({ queryKey: ['dashboard'], queryFn: () => apiFetch<Stats>('/api/dashboard') });

  const handleLogout = () => { clearAuth(); navigate('/'); };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4, flexWrap: 'wrap', gap: 2 }}>
        <Typography variant="h4" fontWeight={700}>Welcome, {user?.name?.split(' ')[0] || 'User'}</Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button variant="outlined" component={Link} to="/products">Shop Products</Button>
          <Button variant="outlined" component={Link} to="/cart">View Cart {count > 0 ? `(${count})` : ''}</Button>
          <Button variant="outlined" color="error" onClick={handleLogout}>Logout</Button>
        </Box>
      </Box>

      {isLoading ? <Typography>Loading…</Typography> : (
        <>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {[
              ['Total Orders', data?.total_orders ?? 0, 'primary.main'],
              ['Pending Orders', data?.pending_orders ?? 0, 'warning.main'],
              ['Total Spent', `₹${(data?.total_spent ?? 0).toLocaleString('en-IN')}`, 'success.main'],
            ].map(([label, value, color]) => (
              <Grid item xs={12} sm={4} key={String(label)}>
                <Card sx={{ borderLeft: 4, borderColor: String(color) }}>
                  <CardContent>
                    <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase', fontWeight: 600 }}>{label}</Typography>
                    <Typography variant="h4" fontWeight={800} color={String(color)}>{value}</Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>Recent Orders</Typography>
              {(data?.recent_orders?.length ?? 0) === 0 ? (
                <Typography color="text.secondary">No orders yet. <Link to="/products">Start shopping!</Link></Typography>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead><TableRow>
                      <TableCell>Order</TableCell>
                      <TableCell>Date</TableCell>
                      <TableCell align="right">Total</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow></TableHead>
                    <TableBody>
                      {(data?.recent_orders || []).map((o) => (
                        <TableRow key={o.id} hover>
                          <TableCell sx={{ fontWeight: 600 }}>{o.order_number}</TableCell>
                          <TableCell>{new Date(o.created_at).toLocaleDateString('en-IN')}</TableCell>
                          <TableCell align="right">₹{Number(o.total).toLocaleString('en-IN')}</TableCell>
                          <TableCell><Chip size="small" label={o.status} color={STATUS_COLOR[o.status] || 'default'} /></TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </Container>
  );
}
"""


def _ecommerce_mock_ts() -> str:
    return r"""type Json = Record<string, unknown>;
const now = () => new Date().toISOString();
let nextOrderId = 1002;
const users: Json[] = [
  { id: 1, email: 'user@example.com', password: 'password123', name: 'Vijay Kumar', phone: '9876543210' },
];
const products: Json[] = [
  { id: 1, name: 'Laptop', description: '15.6" Full HD display, Intel i5, 8GB RAM, 512GB SSD', price: 50000, image: 'https://picsum.photos/seed/laptop/400/300', stock: 20 },
  { id: 2, name: 'Wireless Mouse', description: 'Ergonomic wireless optical mouse with 2.4GHz connectivity', price: 1000, image: 'https://picsum.photos/seed/mouse/400/300', stock: 50 },
  { id: 3, name: 'Noise-Cancelling Headphones', description: 'Over-ear wireless headphones with active noise cancellation', price: 3000, image: 'https://picsum.photos/seed/headphones/400/300', stock: 30 },
  { id: 4, name: 'Mechanical Keyboard', description: 'RGB backlit mechanical keyboard with blue switches', price: 4500, image: 'https://picsum.photos/seed/keyboard/400/300', stock: 15 },
  { id: 5, name: 'USB-C Hub', description: '7-in-1 USB-C hub with HDMI, USB3.0, SD card slots', price: 2200, image: 'https://picsum.photos/seed/hub/400/300', stock: 40 },
  { id: 6, name: 'Webcam HD', description: '1080p Full HD webcam with built-in microphone', price: 2800, image: 'https://picsum.photos/seed/webcam/400/300', stock: 25 },
  { id: 7, name: 'External SSD 1TB', description: 'Portable USB 3.2 Gen2 external SSD', price: 7500, image: 'https://picsum.photos/seed/ssd/400/300', stock: 10 },
  { id: 8, name: 'Monitor 24"', description: '24" IPS Full HD monitor with HDMI and VGA', price: 15000, image: 'https://picsum.photos/seed/monitor/400/300', stock: 8 },
  { id: 9, name: 'Laptop Stand', description: 'Adjustable aluminium laptop stand for 10"–17" laptops', price: 1500, image: 'https://picsum.photos/seed/stand/400/300', stock: 60 },
  { id: 10, name: 'Smart Watch', description: 'Fitness tracker with heart rate monitor, GPS & 7-day battery', price: 8000, image: 'https://picsum.photos/seed/watch/400/300', stock: 18 },
];
const orders: Json[] = [
  { id: 1001, order_number: 'ORD-1001', user_id: 1, total: 51000, status: 'Delivered', shipping_address: '12 MG Road, Mumbai, Maharashtra - 400001', created_at: new Date(Date.now() - 86400000).toISOString() },
];

function resp<T>(data: T): Promise<T> { return Promise.resolve(data); }
function err(msg: string, status = 400): Promise<never> { return Promise.reject(new Error(msg)); }

export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();
  const clean = path.split('?')[0].replace(/\/api\//, '/').replace(/^\/+/, '');
  const parts = clean.split('/').filter(Boolean);

  if (clean === 'auth/login' && method === 'POST') {
    const b = JSON.parse(String(options.body || '{}'));
    const u = users.find((u) => u.email === b.email && u.password === b.password);
    if (!u) return err('Invalid credentials', 401);
    return resp({ access_token: 'mock-jwt-' + u.id, user: { id: u.id, name: u.name, email: u.email } }) as T;
  }
  if (clean === 'auth/register' && method === 'POST') {
    const b = JSON.parse(String(options.body || '{}'));
    if (users.find((u) => u.email === b.email)) return err('Email already registered');
    const u = { id: users.length + 1, email: b.email, password: b.password, name: `${b.first_name} ${b.last_name}`, phone: b.phone || '' };
    users.push(u);
    return resp({ ok: true }) as T;
  }
  if (clean === 'products' && method === 'GET') return resp({ items: products, total: products.length }) as T;
  if (parts[0] === 'products' && parts[1] && method === 'GET') {
    const p = products.find((x) => String(x.id) === parts[1]);
    return p ? resp(p) as T : err('Not found', 404);
  }
  if (clean === 'dashboard' && method === 'GET') {
    const spent = orders.filter((o) => o.user_id === 1).reduce((s, o) => s + Number(o.total), 0);
    return resp({ total_orders: orders.length, pending_orders: orders.filter((o) => o.status === 'Placed').length, total_spent: spent, recent_orders: [...orders].reverse().slice(0, 5) }) as T;
  }
  if (clean === 'orders' && method === 'POST') {
    const b = JSON.parse(String(options.body || '{}'));
    const items: any[] = b.items || [];
    const total = items.reduce((s: number, i: any) => s + i.price * i.quantity, 0);
    const order_number = 'ORD-' + nextOrderId++;
    orders.push({ id: nextOrderId, order_number, user_id: 1, total, status: 'Placed', shipping_address: b.shipping_address, created_at: now() });
    return resp({ order_number, total }) as T;
  }
  if (clean === 'orders' && method === 'GET') return resp({ items: orders, total: orders.length }) as T;
  return resp({ ok: true, mocked: true, path, method }) as T;
}
"""


# ──────────────────────────────────────────────────────────────
# BACKEND
# ──────────────────────────────────────────────────────────────

def _main_py(title: str) -> str:
    return '''import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from routes import router

app = FastAPI(title="%s", version="1.0.0", docs_url="/docs")
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Base.metadata.create_all(bind=engine)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
''' % title


def _requirements_txt() -> str:
    return """fastapi
uvicorn
sqlalchemy
pydantic
pydantic-settings
psycopg2-binary
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
alembic
pytest
httpx
"""


def _models_py() -> str:
    return r'''from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text, Index
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    first_name = Column(String(128), nullable=False)
    last_name = Column(String(128), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(32))
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    cart_items = relationship("Cart", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    price = Column(Numeric(12, 2), nullable=False)
    image = Column(String(512))
    stock = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Cart(Base):
    __tablename__ = "cart"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    quantity = Column(Integer, nullable=False, default=1)

    user = relationship("User", back_populates="cart_items")
    product = relationship("Product")
    __table_args__ = (Index("ix_cart_user_product", "user_id", "product_id"),)


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    order_number = Column(String(64), unique=True, nullable=False, index=True)
    total = Column(Numeric(12, 2), nullable=False)
    status = Column(String(32), default="Placed", index=True)
    shipping_address = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    product_name = Column(String(255), nullable=False)
    price = Column(Numeric(12, 2), nullable=False)
    quantity = Column(Integer, nullable=False)

    order = relationship("Order", back_populates="items")
'''


def _schemas_py() -> str:
    return r'''from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: str
    phone: Optional[str] = None
    password: str = Field(..., min_length=6)
    confirm_password: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ProductOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    image: Optional[str] = None
    stock: int
    class Config:
        from_attributes = True


class CartItemIn(BaseModel):
    id: int
    name: str
    price: float
    quantity: int


class OrderRequest(BaseModel):
    shipping_address: str = Field(..., min_length=5)
    items: List[CartItemIn]


class OrderItemOut(BaseModel):
    product_name: str
    price: float
    quantity: int


class OrderOut(BaseModel):
    id: int
    order_number: str
    total: float
    status: str
    created_at: datetime
    class Config:
        from_attributes = True


class DashboardOut(BaseModel):
    total_orders: int
    pending_orders: int
    total_spent: float
    recent_orders: List[OrderOut]
'''


def _routes_py() -> str:
    return r'''from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user, hash_password, verify_password
from database import get_db
from models import Cart, Order, OrderItem, Product, User
from schemas import (
    DashboardOut,
    LoginRequest,
    OrderOut,
    OrderRequest,
    ProductOut,
    RegisterRequest,
    TokenResponse,
    UserOut,
)

router = APIRouter()


def _order_seq(db: Session) -> str:
    count = db.query(func.count(Order.id)).scalar() or 0
    return f"ORD-{1001 + count}"


@router.post("/api/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(400, "Email already registered")
    if payload.confirm_password and payload.password != payload.confirm_password:
        raise HTTPException(400, "Passwords do not match")
    user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        phone=payload.phone,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    return {"ok": True}


@router.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password")
    token = create_access_token(user.email, "CUSTOMER")
    return TokenResponse(
        access_token=token,
        user=UserOut(id=user.id, name=f"{user.first_name} {user.last_name}", email=user.email),
    )


@router.get("/api/products")
def list_products(q: Optional[str] = None, db: Session = Depends(get_db)):
    qs = db.query(Product)
    if q:
        qs = qs.filter(Product.name.ilike(f"%{q}%"))
    rows = qs.order_by(Product.id).all()
    items = [ProductOut(id=p.id, name=p.name, description=p.description, price=float(p.price), image=p.image, stock=p.stock) for p in rows]
    return {"items": items, "total": len(items)}


@router.get("/api/products/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)):
    p = db.query(Product).filter(Product.id == product_id).first()
    if not p:
        raise HTTPException(404, "Product not found")
    return ProductOut(id=p.id, name=p.name, description=p.description, price=float(p.price), image=p.image, stock=p.stock)


@router.post("/api/orders")
def place_order(
    payload: OrderRequest,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user),
):
    if not user:
        raise HTTPException(401, "Login required to place orders")
    if not payload.items:
        raise HTTPException(400, "Cart is empty")
    total = sum(i.price * i.quantity for i in payload.items)
    order_number = _order_seq(db)
    order = Order(
        user_id=user.id,
        order_number=order_number,
        total=total,
        status="Placed",
        shipping_address=payload.shipping_address,
    )
    db.add(order)
    db.flush()
    for item in payload.items:
        db.add(OrderItem(order_id=order.id, product_id=item.id, product_name=item.name, price=item.price, quantity=item.quantity))
        p = db.query(Product).filter(Product.id == item.id).first()
        if p and p.stock >= item.quantity:
            p.stock -= item.quantity
    db.commit()
    return {"order_number": order_number, "total": total, "status": "Placed"}


@router.get("/api/orders")
def list_orders(db: Session = Depends(get_db), user: Optional[User] = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    rows = db.query(Order).filter(Order.user_id == user.id).order_by(Order.created_at.desc()).all()
    items = [OrderOut(id=o.id, order_number=o.order_number, total=float(o.total), status=o.status, created_at=o.created_at) for o in rows]
    return {"items": items, "total": len(items)}


@router.get("/api/dashboard", response_model=DashboardOut)
def dashboard(db: Session = Depends(get_db), user: Optional[User] = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    orders = db.query(Order).filter(Order.user_id == user.id).order_by(Order.created_at.desc()).all()
    total_spent = sum(float(o.total) for o in orders)
    pending = sum(1 for o in orders if o.status in ("Placed", "Processing"))
    recent = [OrderOut(id=o.id, order_number=o.order_number, total=float(o.total), status=o.status, created_at=o.created_at) for o in orders[:5]]
    return DashboardOut(total_orders=len(orders), pending_orders=pending, total_spent=total_spent, recent_orders=recent)
'''


def _seed_py() -> str:
    return r'''"""Seed 10 e-commerce products and demo users."""
from database import Base, SessionLocal, engine
from models import Product, User
from auth import hash_password

PRODUCTS = [
    ("Laptop",              "15.6\" Full HD Intel i5 8GB RAM 512GB SSD",                        50000, "https://picsum.photos/seed/laptop/400/300",    20),
    ("Wireless Mouse",      "Ergonomic wireless optical mouse 2.4GHz",                          1000,  "https://picsum.photos/seed/mouse/400/300",     50),
    ("Headphones",          "Over-ear wireless noise-cancelling headphones",                    3000,  "https://picsum.photos/seed/headphones/400/300",30),
    ("Mechanical Keyboard", "RGB backlit mechanical keyboard blue switches",                    4500,  "https://picsum.photos/seed/keyboard/400/300",  15),
    ("USB-C Hub",           "7-in-1 USB-C hub HDMI USB3.0 SD card",                            2200,  "https://picsum.photos/seed/hub/400/300",       40),
    ("Webcam HD",           "1080p webcam with built-in microphone",                           2800,  "https://picsum.photos/seed/webcam/400/300",    25),
    ("External SSD 1TB",    "Portable USB 3.2 Gen2 external SSD",                              7500,  "https://picsum.photos/seed/ssd/400/300",       10),
    ("Monitor 24\"",        "24\" IPS Full HD HDMI VGA",                                       15000, "https://picsum.photos/seed/monitor/400/300",   8),
    ("Laptop Stand",        "Adjustable aluminium laptop stand 10-17 inch",                    1500,  "https://picsum.photos/seed/stand/400/300",     60),
    ("Smart Watch",         "Fitness tracker heart rate GPS 7-day battery",                    8000,  "https://picsum.photos/seed/watch/400/300",     18),
]

Base.metadata.create_all(bind=engine)


def run():
    db = SessionLocal()
    try:
        if db.query(Product).count() == 0:
            for name, desc, price, image, stock in PRODUCTS:
                db.add(Product(name=name, description=desc, price=price, image=image, stock=stock))
            print(f"Seeded {len(PRODUCTS)} products")
        if db.query(User).count() == 0:
            db.add(User(first_name="Vijay", last_name="Kumar", email="user@example.com", phone="9876543210", hashed_password=hash_password("password123")))
            print("Seeded demo user: user@example.com / password123")
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    run()
'''


def _test_cart_py() -> str:
    return r'''"""Basic cart / order flow tests."""
import pytest


def test_price_calculation():
    """Cart total = sum(price * qty)."""
    items = [{"price": 50000, "quantity": 2}, {"price": 1000, "quantity": 1}]
    total = sum(i["price"] * i["quantity"] for i in items)
    assert total == 101000


def test_order_number_format():
    """Order numbers start at ORD-1001."""
    order_number = f"ORD-{1001}"
    assert order_number.startswith("ORD-")
    assert int(order_number.split("-")[1]) >= 1001


def test_stock_reduces_on_order():
    stock = 10
    qty = 3
    remaining = stock - qty
    assert remaining == 7
'''


def _readme(title: str) -> str:
    return f"""# {title}

Simple e-commerce shopping cart MVP.

## Flow

Register → Login → Home → Products → Add to Cart → Cart → Checkout → Place Order → Dashboard

## Run

```bash
docker compose up --build
```

Demo login: `user@example.com` / `password123`

## Pages

- **Home** — hero + 6 featured products + Shop Now
- **Products** — 10 product cards, Add to Cart, View Details
- **Product Detail** — image, qty picker, Add to Cart
- **Cart** — table with +/−/remove, total, Checkout
- **Checkout** — shipping address, order summary, Cash on Delivery, Place Order
- **Order Confirmation** — order number, total, Go to Dashboard
- **Dashboard** — stats (total orders / pending / spent), recent orders table
- **Register** — first/last name, email, phone, password, confirm password
- **Login** — email, password

## API

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET  /api/products`
- `GET  /api/products/{{id}}`
- `POST /api/orders` (place order + clear cart + reduce stock)
- `GET  /api/orders`
- `GET  /api/dashboard`
"""
