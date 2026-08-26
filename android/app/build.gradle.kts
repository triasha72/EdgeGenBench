plugins {
    id("com.android.application")
    kotlin("android")
}

val edgegenbenchGitRevision = providers.environmentVariable("GITHUB_SHA").orElse("local").get()
val edgegenbenchQnnEnabled = providers.gradleProperty("edgegenbenchEnableQnn")
    .map(String::toBooleanStrict)
    .orElse(false)
    .get()
val edgegenbenchQnnRoot = providers.gradleProperty("edgegenbenchQnnRoot").orNull

if (edgegenbenchQnnEnabled && edgegenbenchQnnRoot.isNullOrBlank()) {
    error("-PedgegenbenchQnnRoot=/absolute/path is required when QNN is enabled")
}

android {
    namespace = "dev.edgegenbench"
    compileSdk = 35
    ndkVersion = "27.0.12077973"
    defaultConfig {
        applicationId = "dev.edgegenbench"
        minSdk = 28
        targetSdk = 35
        versionCode = 8
        versionName = "0.1.7"
        buildConfigField("String", "GIT_REVISION", "\"$edgegenbenchGitRevision\"")
        buildConfigField("boolean", "QNN_COMPILED", edgegenbenchQnnEnabled.toString())
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON"
                if (edgegenbenchQnnEnabled) {
                    arguments += "-DEDGEBENCH_ENABLE_ORT=ON"
                    arguments += "-DONNXRUNTIME_ROOT=$edgegenbenchQnnRoot"
                }
            }
        }
        ndk {
            abiFilters += if (edgegenbenchQnnEnabled) listOf("arm64-v8a")
            else listOf("arm64-v8a", "x86_64")
        }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    buildFeatures {
        viewBinding = false
        buildConfig = true
    }
    packaging { jniLibs { useLegacyPackaging = false } }
    if (edgegenbenchQnnEnabled) {
        sourceSets.getByName("main").jniLibs.srcDir("$edgegenbenchQnnRoot/lib")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin { jvmToolchain(17) }

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
