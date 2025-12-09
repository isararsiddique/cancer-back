# Dashboard Authentication System

## Overview
The system now enforces strict dashboard separation between Hospital and Researcher dashboards. Users must login separately for each dashboard and cannot switch between them without re-authenticating.

## Changes Made

### 1. Token-Based Dashboard Locking
- JWT tokens now include a `dashboard_type` claim
- Valid values: `"hospital"` or `"researcher"`
- Tokens are locked to the dashboard type specified during login

### 2. Login Endpoint Updated
**Endpoint:** `POST /api/v1/auth/login`

**New Parameter:**
- `dashboard_type` (optional): Specify "hospital" or "researcher" to lock the token

**Example:**
```bash
curl -X POST "http://localhost:5000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=password123&dashboard_type=hospital"
```

### 3. New Dependency: `dashboard_required()`
A new FastAPI dependency that validates the dashboard type from the JWT token.

**Usage in endpoints:**
```python
from core.deps import dashboard_required

@router.get("/patients/")
def list_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _dashboard_check: bool = Depends(dashboard_required("hospital"))
):
    # This endpoint only works with hospital dashboard tokens
    pass
```

### 4. Protected Endpoints

#### Hospital Dashboard Endpoints
All patient-related endpoints now require `dashboard_required("hospital")`:
- `POST /api/v1/patients/` - Create patient
- `GET /api/v1/patients/` - List patients
- `PUT /api/v1/patients/{id}` - Update patient
- `GET /api/v1/patients/export/excel` - Export patients
- And all other patient management endpoints

#### Researcher Dashboard Endpoints
Research-related endpoints require `dashboard_required("researcher")`:
- `GET /api/v1/research/requests/my` - List my research requests
- `POST /api/v1/research/request/create` - Create research request
- And other research-specific endpoints

## Frontend Integration

### Login Flow
1. User selects dashboard type (Hospital or Researcher)
2. Frontend sends login request with `dashboard_type` parameter
3. Backend returns JWT token locked to that dashboard
4. Frontend stores token and uses it for all API calls

### Switching Dashboards
1. User clicks to switch from Hospital to Researcher (or vice versa)
2. Frontend detects dashboard change
3. **Frontend must redirect to login page**
4. User must re-authenticate with new `dashboard_type`
5. Old token is invalidated/ignored

### Error Handling
When a user tries to access a dashboard with wrong token:
- **Status Code:** 401 Unauthorized
- **Error Message:** "This session is for {current_dashboard} dashboard. Please login to access {required_dashboard} dashboard."

**Frontend should:**
1. Catch 401 errors
2. Clear stored tokens
3. Redirect to login page
4. Show message: "Please login to access this dashboard"

## Implementation Example

### Frontend Login Component
```javascript
async function login(email, password, dashboardType) {
  const formData = new FormData();
  formData.append('username', email);
  formData.append('password', password);
  formData.append('dashboard_type', dashboardType); // 'hospital' or 'researcher'
  
  const response = await fetch('/api/v1/auth/login', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('dashboard_type', dashboardType);
}
```

### Frontend Dashboard Switch
```javascript
function switchDashboard(newDashboardType) {
  const currentDashboard = localStorage.getItem('dashboard_type');
  
  if (currentDashboard !== newDashboardType) {
    // Clear tokens
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('dashboard_type');
    
    // Redirect to login
    window.location.href = `/login?dashboard=${newDashboardType}`;
  }
}
```

### Frontend API Error Handler
```javascript
async function apiCall(url, options) {
  const response = await fetch(url, options);
  
  if (response.status === 401) {
    const error = await response.json();
    if (error.detail.includes('dashboard')) {
      // Dashboard mismatch - force re-login
      localStorage.clear();
      window.location.href = '/login';
    }
  }
  
  return response;
}
```

## Security Benefits
1. **Prevents unauthorized dashboard access** - Users can't switch dashboards without re-authentication
2. **Audit trail** - Each dashboard session is tracked separately
3. **Role separation** - Hospital staff can't accidentally access researcher features
4. **Token isolation** - Compromised token only affects one dashboard

## Migration Notes
- Existing tokens without `dashboard_type` will be rejected
- All users must re-login after this update
- Frontend must be updated to send `dashboard_type` parameter during login

## Testing
```bash
# Test hospital login
curl -X POST "http://localhost:5000/api/v1/auth/login" \
  -d "username=doctor@hospital.com&password=pass123&dashboard_type=hospital"

# Test researcher login
curl -X POST "http://localhost:5000/api/v1/auth/login" \
  -d "username=researcher@university.edu&password=pass123&dashboard_type=researcher"

# Test cross-dashboard access (should fail)
# 1. Login as hospital user
# 2. Try to access researcher endpoint - should get 401 error
```
