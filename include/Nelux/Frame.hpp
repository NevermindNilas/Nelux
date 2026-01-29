// Frame.hpp
#pragma once

#include "error/CxException.hpp" // Custom exception class for FFmpeg errors

namespace nelux
{
/**
 * @class Frame
 * @brief A simple RAII wrapper for FFmpeg's AVFrame structure.
 *
 * This class manages the lifecycle of an AVFrame, ensuring proper allocation,
 * cloning, and deallocation. It provides basic accessor methods to interact
 * with the underlying AVFrame data.
 */
class Frame
{
  public:
    /**
     * @brief Default constructor that allocates a new AVFrame.
     *
     * @throws FFException if AVFrame allocation fails.
     */
    Frame();

    /**
     * @brief Constructor that takes an existing AVFrame pointer.
     *
     * @param frame Pointer to an existing AVFrame.
     * @throws FFException if the provided AVFrame pointer is null.
     */
    Frame(AVFrame* frame);

    /**
     * @brief Constructor that allocates a hardware AVFrame using the specified hardware
     * context.
     *
     * @param hw_frames_ctx Pointer to an initialized AVHWFramesContext.
     * @throws FFException if AVFrame allocation or initialization fails.
     */
    Frame(AVBufferRef* hw_frames_ref);


    /**
     * @brief Destructor that frees the AVFrame.
     */
    virtual ~Frame();

    // Copy constructor
    /**
     * @brief Copy constructor that clones the AVFrame from another Frame.
     *
     * @param other The Frame object to copy from.
     * @throws FFException if AVFrame cloning fails.
     */
    Frame(const Frame& other);

    // Copy assignment operator
    /**
     * @brief Copy assignment operator that clones the AVFrame from another Frame.
     *
     * @param other The Frame object to copy from.
     * @return Reference to the assigned Frame object.
     * @throws FFException if AVFrame cloning fails.
     */
    Frame& operator=(const Frame& other);

    // Move constructor
    /**
     * @brief Move constructor that transfers ownership of the AVFrame.
     *
     * @param other The Frame object to move from.
     */
    Frame(Frame&& other) noexcept;

    // Move assignment operator
    /**
     * @brief Move assignment operator that transfers ownership of the AVFrame.
     *
     * @param other The Frame object to move from.
     * @return Reference to the assigned Frame object.
     */
    Frame& operator=(Frame&& other) noexcept;

    /**
     * @brief Access the underlying AVFrame pointer.
     *
     * @return Pointer to the AVFrame.
     */
    AVFrame* get() const;

    /**
     * @brief Get the width of the frame.
     *
     * @return Width in pixels.
     */
    int getWidth() const;

    /**
     * @brief Get the height of the frame.
     *
     * @return Height in pixels.
     */
    int getHeight() const;

    /**
     * @brief Get the pixel format of the frame.
     *
     * @return AVPixelFormat enumeration value.
     */
    AVPixelFormat getPixelFormat() const;

    /**
     * @brief Get Pixel Format as a String
     *
     * @return pixelformat as a string
     *.
     */
    std::string getPixelFormatString() const;

    /**
     * @brief Get the data pointer for a specific plane.
     *
     * @param plane Plane index (0 for Y, 1 for U, 2 for V in YUV formats).
     * @return Pointer to the data of the specified plane.
     * @throws FFException if the plane index is out of range.
     */
    uint8_t* getData(int plane = 0) const;

    /**
     * @brief Get the line size for a specific plane.
     *
     * @param plane Plane index.
     * @return Line size in bytes.
     * @throws FFException if the plane index is out of range.
     */
    int getLineSize(int plane = 0) const;

    /**
     * @brief Get the presentation timestamp of the frame.
     *
     * @return Presentation timestamp (PTS).
     */
    int64_t getPts() const;

    /**
     * @brief Set the presentation timestamp of the frame.
     *
     * @param pts New presentation timestamp.
     */
    void setPts(int64_t pts);

    /**
     * @brief Check if the Frame holds a valid AVFrame.
     *
     * @return True if the AVFrame is valid, False otherwise.
     */
    operator bool() const;

    /**
     * @brief Allocate buffer for the AVFrame with the specified alignment.
     *
     * @param align Alignment in bytes (e.g., 32 for SIMD optimizations).
     * @throws FFException if buffer allocation fails.
     */
    void allocateBuffer(int align = 32);

    /**
     * @brief Copy frame data from another Frame.
     *
     * @param other The source Frame to copy data from.
     * @throws FFException if frame copying fails.
     */
    void copyFrom(const Frame& other);

    /**
     * @brief Fill the frame with raw data.
     *
     * @param data Pointer to the raw data buffer.
     * @param size Size of the data buffer in bytes.
     * @param plane Plane index to fill.
     * @throws FFException if data copying fails or plane index is invalid.
     */
    void fillData(uint8_t* data, int size, int plane = 0);

     /**
     * @brief Get the underlying FFmpeg color space (e.g. AVCOL_SPC_BT709, etc.).
     *
     * If the colorspace is `AVCOL_SPC_UNSPECIFIED`, you might guess one
     * based on resolution or other heuristics.
     *
     * @return AVColorSpace enumeration value.
     */
    AVColorSpace getColorSpace() const;

    /**
     * @brief Get the underlying FFmpeg color range (e.g. AVCOL_RANGE_MPEG,
     * AVCOL_RANGE_JPEG).
     *
     * If the color_range is `AVCOL_RANGE_UNSPECIFIED`, you might guess
     * it is limited or full range. By default, many decoders set `AVCOL_RANGE_MPEG` for
     * 16–235.
     *
     * @return AVColorRange enumeration value.
     */
    AVColorRange getColorRange() const;

    // Optionally, a helper struct if you want everything in one place:
    struct ColorInfo
    {
        AVColorSpace space;
        AVColorRange range;
        // You could add color_primaries, color_trc, etc., if needed
    };

    /**
     * @brief Get or guess the color information for this frame.
     *
     * This method returns a ColorInfo struct containing color space and range.
     */
    ColorInfo getOrGuessColorInfo() const;

    /**
     * @brief Overload the << operator to print Frame information.
     *
     * @param os Output stream.
     * @param frame The Frame object to print.
     * @return Reference to the output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Frame& frame);

  private:
    AVFrame* frame; ///< Pointer to the underlying AVFrame.
};
} // namespace nelux
