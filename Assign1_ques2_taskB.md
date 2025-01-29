# Comparative Analysis of Spectrograms from Different Genres

## Overview
This report presents a comparative analysis of spectrograms for four songs from different genres: Country, EDM, Hip Hop, and Classical. The analysis includes visual comparisons and detailed metrics such as average frequency, peak frequency, spectral centroid, and spectral bandwidth. These metrics provide insights into the unique characteristics of each genre.

## Songs Analyzed
1. **Country:** "Shaboozey - A Bar Song"
2. **EDM:** "Dimitri Vegas"
3. **Hip Hop:** "Desiigner - Panda"
4. **Classical:** "Pachelbel - Canon in D"

## Methodology
1. **Data Collection:** The audio files for the four songs were converted to `.wav` format for compatibility.
2. **Spectrogram Generation:** Using the `librosa` library, spectrograms were generated for each song using the Short-Time Fourier Transform (STFT).
3. **Metrics Calculation:** Metrics such as average frequency, peak frequency, spectral centroid, and spectral bandwidth were computed for each spectrogram.
4. **Visual Comparison:** The spectrograms were visually compared to observe the patterns and frequency content unique to each genre.

## Spectrograms
![TaskB](https://github.com/user-attachments/assets/bb08f4b0-b5a2-42b0-83b3-c32b5af11473)


### Comparative Analysis of Spectrograms from Different Genres

1. **Country (Shaboozey - A Bar Song):**
   - The spectrogram shows a mixture of harmonic content with visible patterns.
   - The frequency content is spread across a wide range, indicating the presence of various instruments such as guitars, drums, and vocals.
   - The harmonic structures are indicative of a typical country music style with clear distinct patterns.
   - **Metrics:**
     - **Average Frequency (Hz):** 11014.24
     - **Peak Frequency (Hz):** 0.00
     - **Spectral Centroid:** 12197.52
     - **Spectral Bandwidth:** 6192.83

2. **EDM (Dimitri Vegas):**
   - The spectrogram shows dense and continuous frequency content, especially in the low to mid-frequency range.
   - The energy is more concentrated, indicating the presence of strong beats and basslines typical of EDM music.
   - The spectrogram also shows periodic bursts of higher frequencies, representing synths and electronic effects.
   - **Metrics:**
     - **Average Frequency (Hz):** 11014.24
     - **Peak Frequency (Hz):** 0.00
     - **Spectral Centroid:** 12310.47
     - **Spectral Bandwidth:** 6240.92

3. **Hip Hop (Desiigner - Panda):**
   - The spectrogram shows a strong presence of low-frequency content, indicative of heavy bass and beats.
   - The mid to high frequencies show less harmonic content and more noise-like structures, typical of percussive elements in hip hop.
   - The vocal elements are visible as vertical lines in the spectrogram, highlighting the rhythmic speech patterns.
   - **Metrics:**
     - **Average Frequency (Hz):** 11014.24
     - **Peak Frequency (Hz):** 817.46
     - **Spectral Centroid:** 12085.22
     - **Spectral Bandwidth:** 6139.33

4. **Classical (Pachelbel - Canon in D):**
   - The spectrogram shows well-defined harmonic structures with rich harmonic content.
   - The frequency content is spread evenly, indicating the presence of multiple instruments such as strings and woodwinds.
   - The harmonic patterns are consistent and repetitive, representing the structured and melodic nature of classical music.
   - **Metrics:**
     - **Average Frequency (Hz):** 11014.24
     - **Peak Frequency (Hz):** 0.00
     - **Spectral Centroid:** 11511.03
     - **Spectral Bandwidth:** 6129.89

## Detailed Comparative Analysis
### Country (Shaboozey - A Bar Song)
- **Harmonic Content:** The spectrogram for the country song shows a mixture of harmonic content with visible patterns. The frequency content is spread across a wide range, indicating the presence of various instruments such as guitars, drums, and vocals.
- **Patterns:** The harmonic structures are indicative of a typical country music style with clear distinct patterns.

### EDM (Dimitri Vegas)
- **Frequency Content:** The spectrogram for the EDM song shows dense and continuous frequency content, especially in the low to mid-frequency range. The energy is more concentrated, indicating the presence of strong beats and basslines typical of EDM music.
- **Periodic Bursts:** The spectrogram also shows periodic bursts of higher frequencies, representing synths and electronic effects.

### Hip Hop (Desiigner - Panda)
- **Low-Frequency Content:** The spectrogram for the hip hop song shows a strong presence of low-frequency content, indicative of heavy bass and beats. The mid to high frequencies show less harmonic content and more noise-like structures, typical of percussive elements in hip hop.
- **Vocal Elements:** The vocal elements are visible as vertical lines in the spectrogram, highlighting the rhythmic speech patterns.

### Classical (Pachelbel - Canon in D)
- **Harmonic Structures:** The spectrogram for the classical piece shows well-defined harmonic structures with rich harmonic content. The frequency content is spread evenly, indicating the presence of multiple instruments such as strings and woodwinds.
- **Melodic Nature:** The harmonic patterns are consistent and repetitive, representing the structured and melodic nature of classical music.

## Conclusion
The spectrograms reveal distinct patterns and characteristics unique to each genre. Country music shows a balanced harmonic structure, EDM demonstrates dense and energetic content, Hip Hop emphasizes low-frequency beats, and Classical music exhibits rich and structured harmonic patterns. These differences are reflected in the computed metrics, which provide quantitative insights into the spectral content of each genre.
