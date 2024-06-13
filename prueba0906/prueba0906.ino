// #include <SD.h>
// #include <SPI.h>
// #include "fpm.h"
// #include <SoftwareSerial.h>

// #define SSpin 10
// SoftwareSerial mySerial(2, 3);
// FPM finger(&mySerial);
// File dataFile;
// size_t bytesRead;

// void setupSDcard() {
//     Serial.println("Initializing SD card...");
//     if (!SD.begin(SSpin)) {
//         Serial.println("Initialization failed!");
//         while (1);
//     }
//     Serial.println("SD Initialization done.");
// }

// void SendFile(String Filename) {
//     char temp[20];
//     Filename.toCharArray(temp, 20);

//     dataFile = SD.open(temp);
//     if (dataFile) {
//         Serial.println("File opened successfully.");
//         ReadTheFile();
//         dataFile.close();
//     } else {
//         Serial.print("Error opening file: ");
//         Serial.println(temp);
//         delay(1000);
//         setupSDcard();
//         return;
//     }
// }

// void ReadTheFile() {
//     const size_t bufferSize = 64;
//     int divBytes = 10;
//     uint8_t buffer[bufferSize];
//     int times = 0;
//     uint8_t expectedLastBlock[] = {0x38, 0x43, 0x6C, 0x16, 0xF7, 0xF6, 0xF6, 0xF6, 0xF6, 0xF6};

//     while (dataFile.available()) {
//         dataFile.read(buffer, bufferSize);
        
//         Serial.print("Reading 10 bytes of BMP file: ");
//         for (int i = 0; i < divBytes; i++) {
//             Serial.print(buffer[i], HEX);
//             Serial.print(" ");
//             buffer[i] += divBytes;
//             FPMStatus status = finger.uploadImage(buffer, bytesRead);
//             if (status != FPMStatus::OK) {
//                 Serial.println("Failed to upload image fragment.");
//                 return;
//             }
//         }
//         Serial.println();
//     }

    
//     // Inicializa el sensor de huellas dactilares
//     if (!finger.begin(FPM_DEFAULT_PASSWORD, FPM_DEFAULT_ADDRESS, NULL)) {
//         Serial.println("Did not find fingerprint sensor :(");
//         return;
//     }
//     Serial.println("Fingerprint sensor found!");

//     // Procesa la imagen con el sensor de huellas dactilares en fragmentos
//     // while (dataFile.available()) {
//     //     size_t bytesRead = dataFile.read(buffer, bufferSize);
//     //     FPMStatus status = finger.uploadImage(buffer, bytesRead);
//     //     if (status != FPMStatus::OK) {
//     //         Serial.println("Failed to upload image fragment.");
//     //         return;
//     //     }
//     // }
//     // Serial.println("Image uploaded successfully.");

//     // Genera el archivo de caracteres a partir de la imagen
//     FPMStatus status = finger.image2Tz(1);
//     if (status != FPMStatus::OK) {
//         Serial.println("Failed to generate character file.");
//         return;
//     }
//     Serial.println("Character file generated successfully.");

//     // Genera una plantilla
//     status = finger.generateTemplate();
//     if (status != FPMStatus::OK) {
//         Serial.println("Failed to generate template.");
//         return;
//     }
//     Serial.println("Template generated successfully.");

//     // Guarda la plantilla en el buffer del sensor
//     status = finger.storeTemplate(1, 1);
//     if (status != FPMStatus::OK) {
//         Serial.println("Failed to store template.");
//         return;
//     }
//     Serial.println("Template stored successfully.");

//     // Descarga la plantilla para comparación
//     status = finger.downloadTemplate(1);
//     if (status != FPMStatus::OK) {
//         Serial.println("Failed to download template.");
//         return;
//     }
//     Serial.println("Template downloaded successfully.");

//     // Compara las huellas dactilares
//     uint16_t score;
//     status = finger.matchTemplatePair(&score);
//     if (status != FPMStatus::OK) {
//         Serial.println("Fingerprints do not match.");
//         return;
//     }
//     Serial.print("Fingerprints match with a score of ");
//     Serial.println(score);
// }

// void setup() {
//     Serial.begin(57600);
//     setupSDcard();
//     SendFile("prueba.bmp");
// }

// void loop() {
// }
#include <SD.h>
#include <SPI.h>
#include "fpm.h"
#include <SoftwareSerial.h>

#define SSpin 10
SoftwareSerial mySerial(2, 3);
FPM finger(&mySerial);
File dataFile;

void setupSDcard() {
    Serial.println("Initializing SD card...");
    if (!SD.begin(SSpin)) {
        Serial.println("Initialization failed!");
        while (1);
    }
    Serial.println("SD Initialization done.");
}

void SendFile(String Filename) {
    char temp[20];
    Filename.toCharArray(temp, 20);

    dataFile = SD.open(temp);
    if (dataFile) {
        Serial.println("File opened successfully.");
        ReadTheFile();
        dataFile.close();
    } else {
        Serial.print("Error opening file: ");
        Serial.println(temp);
        delay(1000);
        setupSDcard();
        return;
    }
}

void ReadTheFile() {
    const size_t bufferSize = 64;
    int divBytes = 10;
    uint8_t buffer[bufferSize];
    size_t bytesRead;

    while (dataFile.available()) {
        // Leer datos del archivo en el buffer
        bytesRead = dataFile.read(buffer, bufferSize);

        Serial.print("Reading ");
        Serial.print(bytesRead);
        Serial.println(" bytes from BMP file.");

        // Verificar si se leyeron datos válidos
        if (bytesRead > 0) {
            // Procesar cada fragmento de imagen con el sensor R307
            for (int i = 0; i < bytesRead; i++) {
                buffer[i] += divBytes; // Modificar el buffer (ejemplo)
            }

            // Subir fragmento de imagen al sensor R307
            FPMStatus status = finger.uploadImage(buffer, bytesRead);
            if (status != FPMStatus::OK) {
                Serial.println("Failed to upload image fragment.");
                return;
            }
        }
    }

    
    // Inicializa el sensor de huellas dactilares
    if (!finger.begin()) {
        Serial.println("Did not find fingerprint sensor :(");
        return;
    }
    Serial.println("Fingerprint sensor found!");

    // Procesa la imagen con el sensor de huellas dactilares en fragmentos
    // while (dataFile.available()) {
    //     size_t bytesRead = dataFile.read(buffer, bufferSize);
    //     FPMStatus status = finger.uploadImage(buffer, bytesRead);
    //     if (status != FPMStatus::OK) {
    //         Serial.println("Failed to upload image fragment.");
    //         return;
    //     }
    // }
    // Serial.println("Image uploaded successfully.");

    // Genera el archivo de caracteres a partir de la imagen
    FPMStatus status = finger.image2Tz(1);
    if (status != FPMStatus::OK) {
        Serial.println("Failed to generate character file.");
        return;
    }
    Serial.println("Character file generated successfully.");

    // Genera una plantilla
    status = finger.generateTemplate();
    if (status != FPMStatus::OK) {
        Serial.println("Failed to generate template.");
        return;
    }
    Serial.println("Template generated successfully.");

    // Guarda la plantilla en el buffer del sensor
    status = finger.storeTemplate(1, 1);
    if (status != FPMStatus::OK) {
        Serial.println("Failed to store template.");
        return;
    }
    Serial.println("Template stored successfully.");

    // Descarga la plantilla para comparación
    status = finger.downloadTemplate(1);
    if (status != FPMStatus::OK) {
        Serial.println("Failed to download template.");
        return;
    }
    Serial.println("Template downloaded successfully.");

    // Compara las huellas dactilares
    uint16_t score;
    status = finger.matchTemplatePair(&score);
    if (status != FPMStatus::OK) {
        Serial.println("Fingerprints do not match.");
        return;
    }
    Serial.print("Fingerprints match with a score of ");
    Serial.println(score);
}

void setup() {
    Serial.begin(57600);
    setupSDcard();
    SendFile("one_bmp.bmp");
}

void loop() {}
