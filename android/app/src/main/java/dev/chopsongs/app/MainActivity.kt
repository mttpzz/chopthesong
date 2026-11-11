package dev.chopsongs.app

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView

/**
 * Single screen: a WebView that loads the public Hugging Face Space.
 * The app holds no API keys and does no search itself.
 */
class MainActivity : Activity() {

    private lateinit var webView: WebView
    private lateinit var progress: ProgressBar
    private lateinit var message: TextView
    private lateinit var retry: Button

    private val handler = Handler(Looper.getMainLooper())
    private val slowStartTimeout = Runnable { showMessage(getString(R.string.waking_up)) }

    private val spaceUri: Uri = Uri.parse(BuildConfig.SPACE_URL)
    private val spaceHost: String? = spaceUri.host

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.web_view)
        progress = findViewById(R.id.progress)
        message = findViewById(R.id.message)
        retry = findViewById(R.id.retry)

        configureWebView()

        retry.setOnClickListener {
            hideMessage()
            loadSpace()
        }

        if (savedInstanceState != null && webView.restoreState(savedInstanceState) != null) {
            progress.visibility = View.GONE
        } else {
            loadSpace()
        }
    }

    private fun configureWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = false
        }

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val uri = request.url
                if (uri.scheme == "https" && uri.host == spaceHost) {
                    return false
                }
                // Anything outside the Space opens in the system browser.
                startActivity(Intent(Intent.ACTION_VIEW, uri))
                return true
            }

            override fun onPageStarted(view: WebView, url: String?, favicon: android.graphics.Bitmap?) {
                progress.visibility = View.VISIBLE
                handler.removeCallbacks(slowStartTimeout)
                handler.postDelayed(slowStartTimeout, SLOW_START_MS)
            }

            override fun onPageFinished(view: WebView, url: String?) {
                handler.removeCallbacks(slowStartTimeout)
                progress.visibility = View.GONE
                hideMessage()
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: WebResourceError
            ) {
                if (request.isForMainFrame) {
                    handler.removeCallbacks(slowStartTimeout)
                    progress.visibility = View.GONE
                    showMessage(getString(R.string.network_error))
                }
            }
        }
    }

    private fun loadSpace() {
        webView.loadUrl(BuildConfig.SPACE_URL)
    }

    private fun showMessage(text: String) {
        message.text = text
        message.visibility = View.VISIBLE
        retry.visibility = View.VISIBLE
    }

    private fun hideMessage() {
        message.visibility = View.GONE
        retry.visibility = View.GONE
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        webView.saveState(outState)
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        handler.removeCallbacks(slowStartTimeout)
        webView.destroy()
        super.onDestroy()
    }

    companion object {
        /** After this long without a finished page, tell the user the Space may be waking up. */
        private const val SLOW_START_MS = 60_000L
    }
}
