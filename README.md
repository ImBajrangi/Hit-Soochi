# Sakha Search Optimization Integration Guide

The Sakha Search Optimization model is exposed via a REST API, allowing it to be used anywhere—from a Flutter mobile app to a Vanilla JS website.

## 🚀 API Endpoint
- **Base URL**: `http://<your-server-ip>:8000`
- **Optimize Endpoint**: `POST /optimize`

### Request Body
```json
{
  "query": "how to reach banke bihari"
}
```

### Response Body
```json
{
  "original": "how to reach banke bihari",
  "optimized": "how to reach banke bihari in Vrindavan",
  "intent": "GENERAL",
  "confidence": "0.35",
  "seo_keywords": ["Vrindavan", "Brij", "reach", "banke", ...]
}
```

---

## 📱 Mobile App (Flutter)
Use the `http` package to call the optimization API before sending the query to your main search engine.

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<String> getOptimizedQuery(String rawQuery) async {
  final response = await http.post(
    Uri.parse('http://YOUR_SERVER_IP:8000/optimize'),
    headers: {"Content-Type": "application/json"},
    body: jsonEncode({"query": rawQuery}),
  );

  if (response.statusCode == 200) {
    return jsonDecode(response.body)['optimized'];
  }
  return rawQuery; // Fallback to raw query
}
```

---

## 🌐 Web (JavaScript)
Enhance your website's search bar or dynamically update meta tags for SEO.

```javascript
async function optimizeSearch(userQuery) {
  const res = await fetch('http://YOUR_SERVER_IP:8000/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: userQuery })
  });
  
  const data = await res.json();
  console.log("Optimized for SEO:", data.optimized);
  console.log("Recommended Keywords:", data.seo_keywords);
  
  return data;
}
```

## 🛠 Deployment
To keep the optimizer running on your server, use a process manager like `pm2`:
```bash
pm2 start "PYTHONPATH=. python Projects/Sakha/api_server.py" --name sakha-optimizer
```
