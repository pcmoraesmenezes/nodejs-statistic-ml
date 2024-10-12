const express = require('express');
const axios = require('axios');
const cors = require('cors');
const cookieParser = require('cookie-parser');

const app = express();
const PORT = 3000;

app.use(cors({ credentials: true, origin: true }));
app.use(express.json());
app.use(cookieParser());

let sessionCookies = '';

function storeSessionCookies(response) {
    const cookies = response.headers['set-cookie'];
    if (cookies) {
        sessionCookies = cookies.join('; ');
    }
}

app.post('/train', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5000/train', req.body, {
            withCredentials: true,
            headers: { Cookie: sessionCookies },
        });
        storeSessionCookies(response);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5000/predict', req.body, {
            withCredentials: true,
            headers: { Cookie: sessionCookies },
        });
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/coefficients', async (req, res) => {
    try {
        const response = await axios.get('http://127.0.0.1:5000/coefficients', {
            withCredentials: true,
            headers: { Cookie: sessionCookies },
        });
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/plot', async (req, res) => {
    try {
        const response = await axios.get('http://127.0.0.1:5000/plot', {
            withCredentials: true,
            headers: { Cookie: sessionCookies },
        });

        res.send(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Node.js server running on http://localhost:${PORT}`);
});
