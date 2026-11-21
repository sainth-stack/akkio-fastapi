import Logo from "../../assets/images/OtamatLogo.svg";
import eye from "../../assets/svg/eye-fill.svg";
import eye2 from "../../assets/svg/eye-slash.svg";
import { useState } from "react";
import { Link } from "react-router-dom";
import { LoadingIndicator } from "../../components/loader";
import './styles.css'
import { useNavigate } from "react-router-dom";
import { message } from "antd";
import { GoogleLogin } from "@react-oauth/google";
import api, { isNetworkError } from "../../utils/api";
import { clearAuthStorage } from "../../utils/auth";

export const Login = () => {
  const [loading, setLoading] = useState(false)
  const [toggle2, setToggle2] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState('')
  const navigate = useNavigate()

  const persistAuth = (data) => {
    const { access_token, user } = data;
    if (access_token) {
      localStorage.setItem('access_token', access_token);
    }
    if (user) {
      localStorage.setItem('user', JSON.stringify(user));
      if (user.email) localStorage.setItem('email', user.email);
      if (user.name) localStorage.setItem('username', user.name);
      if (user.id) localStorage.setItem('_id', String(user.id));
    }
  };

  const handleLogin = async (event) => {
    event.preventDefault();
    try {
      setLoading(true);
      clearAuthStorage();
      const response = await api.post('/auth/login', {
        email,
        password,
        app: 'akkio',
      });
      persistAuth(response.data);
      navigate('/welcome');
    } catch (error) {
      console.error('Login error:', error);
      if (isNetworkError(error)) {
        message.error('Cannot reach the API server (port 8000). Check that the backend is running and firewall allows port 8000.');
      } else {
        message.error('Invalid email or password');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    const idToken = credentialResponse?.credential;
    if (!idToken) {
      message.error('Google login failed.');
      return;
    }
    try {
      setLoading(true);
      clearAuthStorage();
      const response = await api.post('/auth/google-login', {
        id_token: idToken,
        app: 'akkio',
      });
      persistAuth(response.data);
      navigate('/welcome');
    } catch (error) {
      console.error('Google login error:', error);
      message.error('Google login failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container-fluid row m-0 p-0 vh-100">
      <div className="col-md-6 col-xs-12 col-sm-12 text-center pt-lg-5 mt-lg-5">
        <div className="pt-5">
          <img className="logo1" src={Logo} alt="Logo" width={160} height={130} />
        </div>
        <div className="row mt-3">
          <div className="col-md-9 col-lg-9 col-sm-12 col-xs-12 mx-auto">

            <h2 className="mb-5">Login</h2>

            <div className="mb-4 d-flex justify-content-center">
              <GoogleLogin
                onSuccess={handleGoogleSuccess}
                onError={() => message.error('Google login failed. Please try again.')}
                theme="outline"
                size="large"
                width="100%"
              />
            </div>

            <div className="d-flex align-items-center mb-4">
              <div className="flex-grow-1 border-top"></div>
              <div className="mx-3 text-muted small">OR CONTINUE WITH</div>
              <div className="flex-grow-1 border-top"></div>
            </div>

            <form onSubmit={handleLogin} className="pr-lg-5 pl-lg-5">
              <div className="form-group d-flex flex-column" style={{ textAlign: "start" }}>
                <label className="label2 fs13">Email*</label>
                <input
                  style={{ borderRadius: "40px" }}
                  type="email"
                  className="form-control border"
                  id="email"
                  name="email"
                  autoComplete="off"
                  value={email}
                  required
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>

              <div className="form-group d-flex flex-column mt-3" style={{ textAlign: "start" }}>
                <label className="label2 fs13">Password*</label>
                <input
                  style={{ borderRadius: "40px" }}
                  type={toggle2 ? "text" : "password"}
                  className="form-control border"
                  id="password"
                  name="password"
                  value={password}
                  maxLength={16}
                  minLength={8}
                  required
                  onChange={(e) => setPassword(e.target.value)}
                />
                <div className="relative">
                  <img
                    className="eye3"
                    src={toggle2 ? eye2 : eye}
                    onClick={() => setToggle2(!toggle2)}
                    alt="Toggle password visibility"
                  />
                </div>
              </div>
              <div className="d-flex flex-row-reverse mb-4">
                <Link to="#">
                  <span className="fs-12 cursor-pointer">Forgot Password</span>
                </Link>
              </div>
              <button
                className="font-weight-bold text-uppercase w-100 text-white border-0 login2"
                style={{
                  backgroundColor: "#466657",
                  borderRadius: "40px",
                  height: "40px",
                }}
                type={loading ? "button" : "submit"}
                disabled={loading}
              >
                {loading ? "Logging in..." : 'Login'} {loading ? <LoadingIndicator size={"1"} /> : null}
              </button>
            </form>
          </div>
        </div>
      </div>
      <div className="col-md-6 p-0 login-right-panel" aria-hidden />
    </div>
  )
}
