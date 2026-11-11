// Module configuration. Plugin versions are NOT set here: take them from the
// project that Android Studio generates (see android/README.md, step 1).
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.chopsongs.app"
    compileSdk = 37

    defaultConfig {
        applicationId = "dev.chopsongs.app"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"

        // Public Space URL. Replace the placeholder with the real https://<user>-<space>.hf.space
        buildConfigField("String", "SPACE_URL", "\"https://PLACEHOLDER.hf.space\"")
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
