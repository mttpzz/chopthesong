// Module configuration. Plugin versions come from the root build file, which Android Studio
// generates (see android/README.md, step 1). Recent Android Gradle Plugin versions include
// Kotlin support, so the separate Kotlin plugin is not applied here.
plugins {
    id("com.android.application")
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

        // Public app URL (Streamlit Community Cloud).
        buildConfigField("String", "SPACE_URL", "\"https://chopsongs.streamlit.app/\"")
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
